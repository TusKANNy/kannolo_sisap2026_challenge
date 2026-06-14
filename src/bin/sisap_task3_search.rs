use std::io::Write;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use half::f16;
use rayon::prelude::*;

use std::collections::HashSet;

use kannolo::graph::Graph;
use kannolo::hnsw::{EarlyTerminationStrategy, HNSW, HNSWSearchConfiguration};
use kannolo::sisap::{read_gold_knns_h5, read_sparse_csr_h5, write_results_h5};
use vectorium::IndexSerializer;
use vectorium::core::index::Index;
use vectorium::distances::{Distance, DotProduct};
use vectorium::{Dataset, PlainSparseDataset};

#[derive(Debug, Clone, ValueEnum, Default)]
enum EarlyTerminationArg {
    #[default]
    None,
    DistanceAdaptive,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the HDF5 dataset file (used to read the queries).
    #[clap(long, value_parser)]
    h5_file: String,

    /// The HDF5 group containing the CSR query vectors.
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("otest/queries"))]
    query_group: String,

    /// The path of the index to load.
    #[clap(short, long, value_parser)]
    index_file: String,

    /// The number of top-k results to retrieve.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 30)]
    k: usize,

    /// Comma-separated list of ef_search values; one result file is written per value.
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("100"))]
    ef_search: String,

    /// Name of the algorithm, used in the result attributes/filenames.
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("kannolo-hnsw"))]
    algo_name: String,

    /// Directory where the result HDF5 files are written.
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("results/task3"))]
    output_dir: String,

    /// M used at build time (only used to fill the `params` attribute).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    m: usize,

    /// ef_construction used at build time (only used to fill the `params` attribute).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 150)]
    ef_construction: usize,

    /// Early termination strategy for search.
    #[clap(long, value_enum)]
    #[arg(default_value_t = EarlyTerminationArg::None)]
    early_termination: EarlyTerminationArg,

    /// Lambda parameter for the DistanceAdaptive early termination strategy.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 1.0)]
    lambda: f32,

    /// If set, skip writing the per-config result HDF5 files (useful for grid searches).
    #[clap(long, action)]
    skip_h5: bool,

    /// If set, append one row per ef_search value to this TSV file with columns:
    /// ef_search, lambda, Accuracy, Query Time (microsecs), Memory Usage (Bytes),
    /// Building Time (secs). A header is written if the file does not exist yet.
    #[clap(long, value_parser)]
    tsv_output: Option<String>,
}

fn main() {
    let args: Args = Args::parse();

    let buildtime: f64 = std::fs::read_to_string(format!("{}.buildtime", args.index_file))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0.0);

    println!("Loading index from {}", args.index_file);
    let index: HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> =
        <HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .unwrap();

    // If the index was built with `--reorder-egb`, a `<index_file>.permutation` sidecar
    // file maps ground-level/dataset ids back to the original (pre-reordering) ids.
    let ground_inverse_permutation: Option<Vec<usize>> =
        std::fs::read(format!("{}.permutation", args.index_file))
            .ok()
            .map(|bytes| {
                bytes
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as usize)
                    .collect()
            });

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
    let queries_vec: Vec<_> = queries.iter().collect();

    let gt_group = args.query_group.rsplit_once('/').map_or(
        args.query_group.as_str(),
        |(parent, _)| parent,
    );
    let gold = read_gold_knns_h5(&args.h5_file, gt_group).ok();
    if gold.is_none() {
        println!("No gold standard found at group '{gt_group}/knns'; skipping recall report.");
    }

    std::fs::create_dir_all(&args.output_dir).unwrap_or_else(|e| {
        eprintln!("Error creating output directory: {e:?}");
        std::process::exit(1);
    });

    let early_termination = match args.early_termination {
        EarlyTerminationArg::None => EarlyTerminationStrategy::None,
        EarlyTerminationArg::DistanceAdaptive => EarlyTerminationStrategy::DistanceAdaptive {
            lambda: args.lambda,
        },
    };

    let index_size_bytes: u64 = std::fs::metadata(&args.index_file).map(|m| m.len()).unwrap_or(0);

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
                "ef_search\tlambda\tAccuracy\tQuery Time (microsecs)\tMemory Usage (Bytes)\tBuilding Time (secs)"
            )
            .unwrap();
        }
        file
    });

    for ef_search_str in args.ef_search.split(',') {
        let ef_search: usize = ef_search_str
            .trim()
            .parse()
            .expect("invalid ef_search value");
        let config = HNSWSearchConfiguration::default()
            .with_ef_search(ef_search)
            .with_early_termination(early_termination);

        let start_time = Instant::now();
        let results: Vec<_> = queries_vec
            .par_iter()
            .map(|query| index.search(*query, args.k, &config))
            .collect();
        let querytime = start_time.elapsed().as_secs_f64();

        let mut knns = Vec::with_capacity(num_queries * args.k);
        let mut dists = Vec::with_capacity(num_queries * args.k);
        for res in &results {
            let mut found = 0;
            for scored in res.iter().take(args.k) {
                let original_id = match &ground_inverse_permutation {
                    Some(inv) => inv[scored.vector as usize],
                    None => scored.vector as usize,
                };
                knns.push(original_id as i64 + 1);
                dists.push(scored.distance.distance());
                found += 1;
            }
            for _ in found..args.k {
                knns.push(-1);
                dists.push(f32::INFINITY);
            }
        }

        let (et_params, et_suffix) = match args.early_termination {
            EarlyTerminationArg::None => (String::new(), String::new()),
            EarlyTerminationArg::DistanceAdaptive => (
                format!(",earlyTermination=distanceAdaptive,lambda={}", args.lambda),
                format!("_lambda{}", args.lambda),
            ),
        };
        let params = format!(
            "M={},efConstruction={},efSearch={}{et_params}",
            args.m, args.ef_construction, ef_search
        );
        let output_path = format!(
            "{}/{}_M{}_efC{}_efS{}{et_suffix}.h5",
            args.output_dir, args.algo_name, args.m, args.ef_construction, ef_search
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
            .unwrap_or_else(|e| {
                eprintln!("Error writing results file: {e:?}");
                std::process::exit(1);
            });
        }

        let avg_query_time_us = querytime * 1e6 / num_queries as f64;

        let recall = gold.as_ref().map(|(gold_knns, k_gold)| {
            let kk = args.k.min(*k_gold);
            let sum_recall: f64 = (0..num_queries)
                .map(|i| {
                    let pred: HashSet<i64> = knns[i * args.k..i * args.k + kk].iter().copied().collect();
                    let truth: HashSet<i64> =
                        gold_knns[i * k_gold..i * k_gold + kk].iter().copied().collect();
                    pred.intersection(&truth).count() as f64 / kk as f64
                })
                .sum();
            (sum_recall / num_queries as f64, kk)
        });

        if let Some((recall, kk)) = recall {
            println!(
                "ef_search={ef_search}: avg_query_time={avg_query_time_us:.2} us, recall@{kk}={recall:.4}, querytime={querytime:.4}s -> {output_path}"
            );
        } else {
            println!(
                "ef_search={ef_search}: avg_query_time={avg_query_time_us:.2} us, querytime={querytime:.4}s -> {output_path}"
            );
        }

        if let Some(file) = tsv_file.as_mut() {
            let recall_val = recall.map_or(f64::NAN, |(r, _)| r);
            writeln!(
                file,
                "{ef_search}\t{}\t{recall_val:.6}\t{avg_query_time_us:.4}\t{index_size_bytes}\t{buildtime:.6}",
                args.lambda
            )
            .unwrap();
        }
    }
}
