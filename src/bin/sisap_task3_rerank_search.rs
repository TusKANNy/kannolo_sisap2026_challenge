use std::collections::HashSet;
use std::io::Write;
use std::time::Instant;

use clap::Parser;
use half::f16;
use rayon::prelude::*;

use kannolo::graph::Graph;
use kannolo::hnsw::{EarlyTerminationStrategy, HNSW, HNSWSearchConfiguration};
use kannolo::sisap::{read_gold_knns_h5, read_sparse_csr_h5, write_results_h5};
use vectorium::IndexSerializer;
use vectorium::core::rerank_index::RerankIndex;
use vectorium::core::dataset::ScoredVector;
use vectorium::distances::{Distance, DotProduct};
use vectorium::{Dataset, PlainSparseDataset};
use vectorium::vector::SparseVectorView;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(long, value_parser)]
    h5_file: String,

    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("otest/queries"))]
    query_group: String,

    #[clap(short, long, value_parser)]
    index_file: String,

    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 30)]
    k: usize,

    /// Comma-separated ef_search values for the first-stage HNSW.
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("100"))]
    ef_search: String,

    /// Number of candidates retrieved from the first-stage HNSW before reranking.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 100)]
    k_candidates: usize,

    /// Comma-separated list of h values: top-h query components used for first-stage search.
    /// Full query is always used for reranking.
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("5,10,15"))]
    query_top_h: String,

    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("kannolo-hnsw-rerank"))]
    algo_name: String,

    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("results/task3"))]
    output_dir: String,

    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    m: usize,

    #[clap(long, value_parser)]
    #[arg(default_value_t = 150)]
    ef_construction: usize,

    /// L1 fraction used at build time (only used for the params attribute/filename).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0.8)]
    l1_fraction: f32,

    /// Comma-separated lambda values for distance-adaptive early termination in the first-stage HNSW.
    /// Use "none" to disable. E.g. "none,0.05,0.09,0.13".
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("none"))]
    lambda: String,

    /// Comma-separated alpha values for candidate pruning before reranking.
    /// Use "none" to disable. E.g. "none,0.3,0.4,0.5".
    /// Alpha prunes first-stage candidates whose score is below
    /// (1 - alpha) * score_of_kth_first_stage_result before reranking.
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("none"))]
    alpha: String,

    #[clap(long, action)]
    skip_h5: bool,

    #[clap(long, value_parser)]
    tsv_output: Option<String>,
}

/// Returns owned (components, values) for the top-h query components by value.
/// Re-sorted by component id so the resulting view is compatible with merge-based dot product.
fn prune_query_top_h(components: &[u16], values: &[f32], h: usize) -> (Vec<u16>, Vec<f32>) {
    if h >= components.len() {
        return (components.to_vec(), values.to_vec());
    }
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]));
    order.truncate(h);
    order.sort_unstable();
    let comps = order.iter().map(|&i| components[i]).collect();
    let vals = order.iter().map(|&i| values[i]).collect();
    (comps, vals)
}

fn main() {
    let args: Args = Args::parse();

    let buildtime: f64 = std::fs::read_to_string(format!("{}.buildtime", args.index_file))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0.0);

    println!("Loading first-stage index from {}", args.index_file);
    let first_stage: HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> =
        <HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .unwrap_or_else(|e| {
            eprintln!("Error loading first-stage index: {e:?}");
            std::process::exit(1);
        });

    let full_dataset_path = format!("{}.full", args.index_file);
    println!("Loading full rerank dataset from {}", full_dataset_path);
    let rerank_dataset: PlainSparseDataset<u16, f16, DotProduct> =
        <PlainSparseDataset<u16, f16, DotProduct> as IndexSerializer>::load_index(
            &full_dataset_path,
        )
        .unwrap_or_else(|e| {
            eprintln!("Error loading rerank dataset: {e:?}");
            std::process::exit(1);
        });

    let rerank_index: RerankIndex<
        HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>,
        PlainSparseDataset<u16, f16, DotProduct>,
        PlainSparseDataset<u16, f16, DotProduct>,
    > = RerankIndex::new(first_stage, rerank_dataset);

    println!(
        "Reading queries from {} (group: {})",
        args.h5_file, args.query_group
    );
    let queries: PlainSparseDataset<u16, f32, DotProduct> =
        read_sparse_csr_h5::<f32, DotProduct>(&args.h5_file, &args.query_group).unwrap_or_else(
            |e| {
                eprintln!("Error reading queries: {e:?}");
                std::process::exit(1);
            },
        );
    let num_queries = queries.len();
    let queries_vec: Vec<SparseVectorView<'_, u16, f32>> = queries.iter().collect();

    let gt_group = args
        .query_group
        .rsplit_once('/')
        .map_or(args.query_group.as_str(), |(parent, _)| parent);
    let gold = read_gold_knns_h5(&args.h5_file, gt_group).ok();

    std::fs::create_dir_all(&args.output_dir).unwrap_or_else(|e| {
        eprintln!("Error creating output directory: {e:?}");
        std::process::exit(1);
    });

    let ef_search_values: Vec<usize> = args
        .ef_search
        .split(',')
        .map(|s| s.trim().parse().expect("invalid ef_search value"))
        .collect();

    let h_values: Vec<usize> = args
        .query_top_h
        .split(',')
        .map(|s| s.trim().parse().expect("invalid query_top_h value"))
        .collect();

    let lambda_values: Vec<Option<f32>> = args
        .lambda
        .split(',')
        .map(|s| {
            let s = s.trim();
            if s == "none" { None } else { Some(s.parse::<f32>().expect("invalid lambda value")) }
        })
        .collect();

    let alpha_values: Vec<Option<f32>> = args
        .alpha
        .split(',')
        .map(|s| {
            let s = s.trim();
            if s == "none" { None } else { Some(s.parse::<f32>().expect("invalid alpha value")) }
        })
        .collect();

    let index_size_bytes: u64 =
        std::fs::metadata(&args.index_file).map(|m| m.len()).unwrap_or(0);

    let mut tsv_file = args.tsv_output.as_ref().map(|path| {
        let write_header = !std::path::Path::new(path).exists();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap_or_else(|e| {
                eprintln!("Error opening TSV output file: {e:?}");
                std::process::exit(1);
            });
        if write_header {
            writeln!(
                file,
                "ef_search\tk_candidates\tquery_top_h\tlambda\talpha\tAccuracy\tQuery Time (microsecs)\tMemory Usage (Bytes)\tBuilding Time (secs)"
            )
            .unwrap();
        }
        file
    });

    for &h in &h_values {
        // Pre-compute pruned queries for this h value (avoids recomputing inside par_iter)
        let pruned_queries: Vec<(Vec<u16>, Vec<f32>)> = queries_vec
            .iter()
            .map(|q| prune_query_top_h(q.components(), q.values(), h))
            .collect();

        for &ef_search in &ef_search_values {
            for &lambda in &lambda_values {
            let et = match lambda {
                None => EarlyTerminationStrategy::None,
                Some(l) => EarlyTerminationStrategy::DistanceAdaptive { lambda: l },
            };
            let config = HNSWSearchConfiguration::default()
                .with_ef_search(ef_search)
                .with_early_termination(et);

            for &alpha in &alpha_values {
                let start_time = Instant::now();
                let results: Vec<_> = pruned_queries
                    .par_iter()
                    .zip(queries_vec.par_iter())
                    .map(|((pruned_comps, pruned_vals), full_query)| {
                        let pruned_view =
                            SparseVectorView::new(pruned_comps.as_slice(), pruned_vals.as_slice());
                        rerank_index.search(
                            pruned_view,
                            *full_query,
                            args.k_candidates,
                            args.k,
                            &config,
                            alpha,
                            None,
                        )
                    })
                    .collect();
                let querytime = start_time.elapsed().as_secs_f64();

                let mut knns = Vec::with_capacity(num_queries * args.k);
                let mut dists = Vec::with_capacity(num_queries * args.k);
                for res in &results {
                    let mut found = 0;
                    for scored in res.iter().take(args.k).collect::<Vec<&ScoredVector<DotProduct>>>() {
                        knns.push(scored.vector as i64 + 1);
                        dists.push(scored.distance.distance());
                        found += 1;
                    }
                    for _ in found..args.k {
                        knns.push(-1);
                        dists.push(f32::INFINITY);
                    }
                }

                let alpha_str = alpha.map_or("none".to_string(), |a| format!("{a}"));
                let lambda_str = lambda.map_or("none".to_string(), |l| format!("{l}"));
                let params = format!(
                    "M={},efConstruction={},efSearch={},kCandidates={},queryTopH={},l1Fraction={},lambda={},alpha={}",
                    args.m, args.ef_construction, ef_search, args.k_candidates, h, args.l1_fraction, lambda_str, alpha_str
                );
                let output_path = format!(
                    "{}/{}_M{}_efC{}_efS{}_kC{}_h{}_l{}_a{}.h5",
                    args.output_dir,
                    args.algo_name,
                    args.m,
                    args.ef_construction,
                    ef_search,
                    args.k_candidates,
                    h,
                    lambda_str,
                    alpha_str,
                );

                if !args.skip_h5 {
                    write_results_h5(
                        &output_path,
                        &knns,
                        &dists,
                        num_queries,
                        args.k,
                        &args.algo_name,
                        "task3",
                        buildtime,
                        querytime,
                        &params,
                    )
                    .unwrap_or_else(|e| eprintln!("Error writing results: {e:?}"));
                }

                let avg_query_time_us = querytime * 1e6 / num_queries as f64;

                let recall = gold.as_ref().map(|(gold_knns, k_gold)| {
                    let kk = args.k.min(*k_gold);
                    let sum_recall: f64 = (0..num_queries)
                        .map(|i| {
                            let pred: HashSet<i64> =
                                knns[i * args.k..i * args.k + kk].iter().copied().collect();
                            let truth: HashSet<i64> =
                                gold_knns[i * k_gold..i * k_gold + kk].iter().copied().collect();
                            pred.intersection(&truth).count() as f64 / kk as f64
                        })
                        .sum();
                    (sum_recall / num_queries as f64, kk)
                });

                if let Some((recall, kk)) = recall {
                    println!(
                        "ef={ef_search} kC={} h={h} l={lambda_str} a={alpha_str}: avg_qt={avg_query_time_us:.2} us, recall@{kk}={recall:.4}, qt={querytime:.4}s",
                        args.k_candidates
                    );
                } else {
                    println!(
                        "ef={ef_search} kC={} h={h} l={lambda_str} a={alpha_str}: avg_qt={avg_query_time_us:.2} us, qt={querytime:.4}s",
                        args.k_candidates
                    );
                }

                if let Some(file) = tsv_file.as_mut() {
                    let recall_val = recall.map_or(f64::NAN, |(r, _)| r);
                    writeln!(
                        file,
                        "{ef_search}\t{}\t{h}\t{lambda_str}\t{alpha_str}\t{recall_val:.6}\t{avg_query_time_us:.4}\t{index_size_bytes}\t{buildtime:.6}",
                        args.k_candidates
                    )
                    .unwrap();
                }
            } // alpha
            } // lambda
        } // ef_search
    } // h
}
