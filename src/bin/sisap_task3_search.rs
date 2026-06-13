use std::time::Instant;

use clap::Parser;
use half::f16;
use rayon::prelude::*;

use std::collections::HashSet;

use kannolo::graph::Graph;
use kannolo::hnsw::{HNSW, HNSWSearchConfiguration};
use kannolo::sisap::{read_gold_knns_h5, read_sparse_csr_h5, write_results_h5};
use vectorium::IndexSerializer;
use vectorium::core::index::Index;
use vectorium::distances::{Distance, DotProduct};
use vectorium::{Dataset, PlainSparseDataset};

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

    for ef_search_str in args.ef_search.split(',') {
        let ef_search: usize = ef_search_str
            .trim()
            .parse()
            .expect("invalid ef_search value");
        let config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

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
                knns.push(scored.vector as i64 + 1);
                dists.push(scored.distance.distance());
                found += 1;
            }
            for _ in found..args.k {
                knns.push(-1);
                dists.push(f32::INFINITY);
            }
        }

        let params = format!(
            "M={},efConstruction={},efSearch={}",
            args.m, args.ef_construction, ef_search
        );
        let output_path = format!(
            "{}/{}_M{}_efC{}_efS{}.h5",
            args.output_dir, args.algo_name, args.m, args.ef_construction, ef_search
        );

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

        let avg_query_time_us = querytime * 1e6 / num_queries as f64;

        if let Some((gold_knns, k_gold)) = &gold {
            let kk = args.k.min(*k_gold);
            let sum_recall: f64 = (0..num_queries)
                .map(|i| {
                    let pred: HashSet<i64> = knns[i * args.k..i * args.k + kk].iter().copied().collect();
                    let truth: HashSet<i64> =
                        gold_knns[i * k_gold..i * k_gold + kk].iter().copied().collect();
                    pred.intersection(&truth).count() as f64 / kk as f64
                })
                .sum();
            let recall = sum_recall / num_queries as f64;
            println!(
                "ef_search={ef_search}: avg_query_time={avg_query_time_us:.2} us, recall@{kk}={recall:.4}, querytime={querytime:.4}s -> {output_path}"
            );
        } else {
            println!(
                "ef_search={ef_search}: avg_query_time={avg_query_time_us:.2} us, querytime={querytime:.4}s -> {output_path}"
            );
        }
    }
}
