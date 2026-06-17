use std::time::Instant;

use clap::Parser;
use half::f16;

use kannolo::graph::Graph;
use kannolo::hnsw::{HNSW, HNSWBuildConfiguration};
use kannolo::sisap::read_sparse_csr_h5;
use vectorium::IndexSerializer;
use vectorium::{Dataset, DatasetGrowable, PlainSparseDataset, PlainSparseDatasetGrowable};
use vectorium::core::index::Index;
use vectorium::distances::DotProduct;
use vectorium::vector::SparseVectorView;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(long, value_parser)]
    h5_file: String,

    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("train"))]
    group: String,

    #[clap(short, long, value_parser)]
    output_file: String,

    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    m: usize,

    #[clap(long, value_parser)]
    #[arg(default_value_t = 150)]
    ef_construction: usize,

    /// Fraction of L1 norm to retain per document vector (e.g. 0.8 keeps the top
    /// components summing to 80% of the vector's L1 norm). Values in (0, 1].
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0.8)]
    l1_fraction: f32,
}

/// Returns the indices (into the sorted-by-component-id slice) of the components to keep
/// so that their L1 mass >= `fraction` of the vector's total L1.
/// The returned indices are re-sorted by component position so the resulting view
/// remains sorted by component id (required for sparse merge operations).
fn prune_l1_fraction_indices(values: &[f16], fraction: f32) -> Vec<usize> {
    if fraction >= 1.0 || values.is_empty() {
        return (0..values.len()).collect();
    }
    let total_l1: f32 = values.iter().map(|v| f32::from(*v)).sum();
    let threshold = fraction * total_l1;

    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        f32::from(values[b]).total_cmp(&f32::from(values[a]))
    });

    let mut cumsum = 0.0f32;
    let mut cutoff = order.len();
    for (i, &idx) in order.iter().enumerate() {
        cumsum += f32::from(values[idx]);
        if cumsum >= threshold {
            cutoff = i + 1;
            break;
        }
    }

    let mut kept = order[..cutoff].to_vec();
    kept.sort_unstable();
    kept
}

fn main() {
    let args: Args = Args::parse();

    println!(
        "Reading dataset from {} (group: {})",
        args.h5_file, args.group
    );

    let start_time = Instant::now();

    let full_dataset: PlainSparseDataset<u16, f16, DotProduct> =
        read_sparse_csr_h5::<f16, DotProduct>(&args.h5_file, &args.group).unwrap_or_else(|e| {
            eprintln!("Error reading HDF5 dataset: {e:?}");
            std::process::exit(1);
        });

    println!(
        "Read {} vectors. Building pruned dataset (L1 fraction = {})...",
        full_dataset.len(),
        args.l1_fraction
    );

    // Build pruned first-stage dataset from full dataset
    let mut pruned_growable =
        PlainSparseDatasetGrowable::<u16, f16, DotProduct>::new(full_dataset.encoder().clone());

    let mut total_orig_nnz: usize = 0;
    let mut total_pruned_nnz: usize = 0;
    for vec in full_dataset.iter() {
        let kept = prune_l1_fraction_indices(vec.values(), args.l1_fraction);
        total_orig_nnz += vec.values().len();
        total_pruned_nnz += kept.len();
        let comps: Vec<u16> = kept.iter().map(|&i| vec.components()[i]).collect();
        let vals: Vec<f16> = kept.iter().map(|&i| vec.values()[i]).collect();
        pruned_growable.push(SparseVectorView::new(&comps, &vals));
    }
    println!(
        "Pruned: avg nnz {:.1} -> {:.1} ({:.1}% retained)",
        total_orig_nnz as f64 / full_dataset.len() as f64,
        total_pruned_nnz as f64 / full_dataset.len() as f64,
        100.0 * total_pruned_nnz as f64 / total_orig_nnz as f64,
    );

    let pruned_dataset: PlainSparseDataset<u16, f16, DotProduct> = pruned_growable.into();

    println!(
        "Building first-stage HNSW with M={}, ef_construction={}",
        args.m, args.ef_construction
    );
    let config = HNSWBuildConfiguration::default()
        .with_num_neighbors(args.m)
        .with_ef_construction(args.ef_construction);
    let index: HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> =
        HNSW::build_index(pruned_dataset, &config);

    let build_time = start_time.elapsed();
    println!("Time to build (load + prune + index): {} s", build_time.as_secs_f64());

    index.save_index(&args.output_file).unwrap_or_else(|e| {
        eprintln!("Error saving first-stage index: {e:?}");
        std::process::exit(1);
    });

    let full_dataset_path = format!("{}.full", args.output_file);
    full_dataset.save_index(&full_dataset_path).unwrap_or_else(|e| {
        eprintln!("Error saving full dataset: {e:?}");
        std::process::exit(1);
    });

    std::fs::write(
        format!("{}.buildtime", args.output_file),
        build_time.as_secs_f64().to_string(),
    )
    .unwrap_or_else(|e| eprintln!("Warning: could not write buildtime sidecar: {e:?}"));

    println!(
        "Saved first-stage index -> {}\nSaved full rerank dataset -> {}.full",
        args.output_file, args.output_file
    );
}
