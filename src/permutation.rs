//! Graph reordering via Recursive Graph Bisection (EGB).
//!
//! This module computes a permutation of vector IDs that improves the locality of
//! a graph's adjacency lists (and, if the dataset is reordered with the same
//! permutation, of the dataset itself), then provides helpers to apply that
//! permutation to an `HNSW` index's graph levels.

#[cfg(feature = "egb")]
use rayon::prelude::*;
#[cfg(feature = "egb")]
use rgb::forward::Doc;
#[cfg(feature = "egb")]
use rgb::recursive_graph_bisection;
use std::collections::HashMap;

use crate::graph::GraphTrait;

#[cfg(feature = "egb")]
const ITERATIONS: usize = 10;
#[cfg(feature = "egb")]
const MIN_PARTITION_SIZE: usize = 64;
#[cfg(feature = "egb")]
const MAX_DEPTH: usize = 100;
#[cfg(feature = "egb")]
const PARALLEL_SWITCH: usize = 100;
#[cfg(feature = "egb")]
const DEPTH_LIMIT: usize = 1;
#[cfg(feature = "egb")]
const SORT_LEAF: bool = true;
#[cfg(feature = "egb")]
const ID: usize = 1;

/// Inverts a permutation. Given `p` where `p[old_id] = new_id`, returns `q` where
/// `q[new_id] = old_id`. The operation is self-inverse.
pub fn invert_mapping(p: &[usize]) -> Vec<usize> {
    let n = p.len();
    let mut inv = vec![0usize; n];
    for (i, &j) in p.iter().enumerate() {
        inv[j] = i;
    }
    inv
}

/// Validates that `p` is a permutation of `0..p.len()`.
pub fn validate_permutation(p: &[usize]) -> Result<(), String> {
    let n = p.len();
    let mut seen = vec![false; n];
    for (i, &x) in p.iter().enumerate() {
        if x >= n {
            return Err(format!(
                "invalid permutation: value {x} at position {i} out of range"
            ));
        }
        if seen[x] {
            return Err(format!("invalid permutation: duplicate value {x}"));
        }
        seen[x] = true;
    }
    Ok(())
}

/// Computes an EGB (recursive graph bisection) permutation `old_id -> new_id` from
/// the adjacency lists of `graph`.
#[cfg(feature = "egb")]
pub fn compute_egb_permutation<G>(graph: &G) -> Vec<usize>
where
    G: GraphTrait + Sync,
{
    let n = graph.n_nodes();

    // Collect and compact the set of ids that appear as neighbors, so RGB can work
    // over a dense term-id space.
    let mut all_terms: Vec<u32> = (0..n)
        .into_par_iter()
        .flat_map_iter(|u| graph.neighbors(u).map(|v| v as u32))
        .collect();
    all_terms.par_sort_unstable();
    all_terms.dedup();
    let num_terms = all_terms.len();

    let mut term_map = vec![u32::MAX; n];
    for (new_id, &old_id) in all_terms.iter().enumerate() {
        term_map[old_id as usize] = new_id as u32;
    }

    let mut docs: Vec<Doc> = (0..n)
        .into_par_iter()
        .map(|u| {
            let mut terms: Vec<u32> = graph.neighbors(u).map(|v| term_map[v]).collect();
            terms.sort_unstable();
            terms.dedup();

            Doc {
                terms,
                org_id: u as u32,
                gain: 0.0,
                leaf_id: -1,
            }
        })
        .collect();

    recursive_graph_bisection(
        &mut docs,
        num_terms,
        ITERATIONS,
        MIN_PARTITION_SIZE,
        MAX_DEPTH,
        PARALLEL_SWITCH,
        DEPTH_LIMIT,
        SORT_LEAF,
        ID,
    );

    let mut perm = vec![0usize; n];
    for (new_id, doc) in docs.iter().enumerate() {
        perm[doc.org_id as usize] = new_id;
    }
    perm
}

/// Given an upper HNSW level and the global `old_id -> new_id` permutation, returns
/// the old local ids of `level` listed in the order their corresponding global ids
/// will appear in the new ground-level ordering.
///
/// `previous_old_globals_in_new_order` is the analogous list for the level directly
/// below (already in new order); since HNSW upper levels are nested prefixes of the
/// level below, every global id in it must also appear in `level`, and is placed
/// first (preserving its relative order) so the "is a prefix of" invariant holds in
/// the new ordering too.
pub fn upper_level_order_preserving_hnsw_prefixes<G>(
    level: &G,
    permutation: &[usize],
    previous_old_globals_in_new_order: &[usize],
) -> Vec<usize>
where
    G: GraphTrait,
{
    let n = level.n_nodes();
    let mut old_local_by_old_global = HashMap::with_capacity(n);
    for old_local in 0..n {
        old_local_by_old_global.insert(level.get_external_id(old_local), old_local);
    }

    let mut old_locals_by_new_local = Vec::with_capacity(n);
    let mut used = vec![false; n];

    for &old_global in previous_old_globals_in_new_order {
        let old_local = *old_local_by_old_global
            .get(&old_global)
            .expect("upper HNSW levels must be nested prefixes");
        old_locals_by_new_local.push(old_local);
        used[old_local] = true;
    }

    let mut remaining = (0..n)
        .filter(|&old_local| !used[old_local])
        .collect::<Vec<_>>();
    remaining.sort_unstable_by_key(|&old_local| {
        let old_global = level.get_external_id(old_local);
        permutation[old_global]
    });
    old_locals_by_new_local.extend(remaining);

    old_locals_by_new_local
}
