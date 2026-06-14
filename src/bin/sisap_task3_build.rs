use std::time::Instant;

use clap::Parser;
use half::f16;

use kannolo::graph::Graph;
use kannolo::hnsw::{HNSW, HNSWBuildConfiguration};
use kannolo::permutation::invert_mapping;
use kannolo::sisap::read_sparse_csr_h5;
use vectorium::IndexSerializer;
use vectorium::{Dataset, DatasetGrowable, PlainSparseDataset, PlainSparseDatasetGrowable, VectorId};
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

    /// Reorder the ground level and dataset using EGB graph bisection for better
    /// cache locality during search. Experimental.
    #[clap(long, action)]
    reorder_egb: bool,
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

    let (index, ground_inverse_permutation) = if args.reorder_egb {
        println!("Reordering index with EGB graph bisection...");
        let reorder_start = Instant::now();
        let (reordered, ground_inv) = index.reorder_by_egb(|dataset, permutation| {
            let old_id_by_new_id = invert_mapping(permutation);
            let mut permuted =
                PlainSparseDatasetGrowable::<u16, f16, DotProduct>::new(dataset.encoder().clone());
            for old_id in old_id_by_new_id {
                permuted.push(dataset.get(old_id as VectorId));
            }
            permuted.into()
        });
        println!(
            "Time to reorder with EGB: {} s",
            reorder_start.elapsed().as_secs_f64()
        );
        (reordered, Some(ground_inv))
    } else {
        (index, None)
    };

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

    if let Some(ground_inv) = ground_inverse_permutation {
        let permutation_path = format!("{}.permutation", args.output_file);
        let bytes: Vec<u8> = ground_inv
            .iter()
            .flat_map(|&id| (id as u64).to_le_bytes())
            .collect();
        std::fs::write(&permutation_path, bytes).unwrap_or_else(|e| {
            eprintln!("Warning: could not write permutation sidecar file: {e:?}");
        });
    }
}
