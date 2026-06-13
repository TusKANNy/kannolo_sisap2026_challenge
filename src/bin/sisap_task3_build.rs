use std::time::Instant;

use clap::Parser;
use half::f16;

use kannolo::graph::Graph;
use kannolo::hnsw::{HNSW, HNSWBuildConfiguration};
use kannolo::sisap::read_sparse_csr_h5;
use vectorium::IndexSerializer;
use vectorium::PlainSparseDataset;
use vectorium::core::index::Index;
use vectorium::distances::DotProduct;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the HDF5 dataset file.
    #[clap(long, value_parser)]
    h5_file: String,

    /// The HDF5 group containing the CSR data to index (e.g. "train").
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("train"))]
    group: String,

    /// The output file where to save the index.
    #[clap(short, long, value_parser)]
    output_file: String,

    /// The number of neighbors per node.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    m: usize,

    /// The size of the candidate pool at construction time.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 150)]
    ef_construction: usize,
}

fn main() {
    let args: Args = Args::parse();

    let config = HNSWBuildConfiguration::default()
        .with_num_neighbors(args.m)
        .with_ef_construction(args.ef_construction);

    println!(
        "Reading dataset from {} (group: {})",
        args.h5_file, args.group
    );

    let start_time = Instant::now();

    let dataset: PlainSparseDataset<u16, f16, DotProduct> =
        read_sparse_csr_h5::<f16, DotProduct>(&args.h5_file, &args.group).unwrap_or_else(|e| {
            eprintln!("Error reading HDF5 dataset: {e:?}");
            std::process::exit(1);
        });

    println!(
        "Building Index with M: {}, ef_construction: {}",
        args.m, args.ef_construction
    );

    let index: HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> =
        HNSW::build_index(dataset, &config);

    let build_time = start_time.elapsed();
    println!(
        "Time to build index (load + build): {} s",
        build_time.as_secs_f64()
    );

    let _ = index.save_index(&args.output_file);

    let buildtime_path = format!("{}.buildtime", args.output_file);
    std::fs::write(&buildtime_path, build_time.as_secs_f64().to_string()).unwrap_or_else(|e| {
        eprintln!("Warning: could not write buildtime sidecar file: {e:?}");
    });
}
