//! HDF5 I/O helpers for the SISAP2026 Task 3 (sparse indexing) binaries.

use std::str::FromStr;

use hdf5::File;
use hdf5::types::VarLenUnicode;
use vectorium::DatasetGrowable;
use vectorium::Float;
use vectorium::FromF32;
use vectorium::PlainSparseDataset;
use vectorium::PlainSparseDatasetGrowable;
use vectorium::ValueType;
use vectorium::encoders::sparse_scalar::{PlainSparseQuantizer, ScalarSparseSupportedDistance};
use vectorium::vector::SparseVectorView;

/// Reads a CSR sparse matrix group (`data`/`indices`/`indptr`/`shape`) from an HDF5 file
/// and builds a `PlainSparseDataset` directly from it.
pub fn read_sparse_csr_h5<V, D>(
    path: &str,
    group_name: &str,
) -> hdf5::Result<PlainSparseDataset<u16, V, D>>
where
    V: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    let file = File::open(path)?;
    let group = file.group(group_name)?;

    let shape = group.attr("shape")?.read_raw::<i64>()?;
    let dim = shape[1] as usize;

    let indptr = group.dataset("indptr")?.read_raw::<i64>()?;
    let indices = group.dataset("indices")?.read_raw::<i64>()?;
    let data = group.dataset("data")?.read_raw::<f32>()?;

    let components: Vec<u16> = indices.iter().map(|&c| c as u16).collect();
    let values: Vec<V> = data.iter().map(|&v| V::from_f32_saturating(v)).collect();

    let encoder = PlainSparseQuantizer::<u16, V, D>::new(dim, dim);
    let mut dataset: PlainSparseDatasetGrowable<u16, V, D> = DatasetGrowable::new(encoder);

    for w in indptr.windows(2) {
        let start = w[0] as usize;
        let end = w[1] as usize;
        let view = SparseVectorView::new(&components[start..end], &values[start..end]);
        dataset.push(view);
    }

    Ok(dataset.into())
}

/// Reads the gold-standard `knns` dataset (1-based ids) from `<group_name>/knns`,
/// returning the flat row-major ids together with the number of columns (k_gold).
pub fn read_gold_knns_h5(path: &str, group_name: &str) -> hdf5::Result<(Vec<i64>, usize)> {
    let file = File::open(path)?;
    let dataset = file.dataset(&format!("{group_name}/knns"))?;
    let k_gold = dataset.shape()[1];
    let knns: Vec<i64> = dataset.read_raw::<i32>()?.iter().map(|&x| x as i64).collect();
    Ok((knns, k_gold))
}

/// Writes the SISAP2026 Task 3 result file: an n x k `knns` matrix (1-based ids) and
/// `dists` matrix, plus the required root attributes.
#[allow(clippy::too_many_arguments)]
pub fn write_results_h5(
    path: &str,
    knns: &[i64],
    dists: &[f32],
    n_queries: usize,
    k: usize,
    algo: &str,
    task: &str,
    buildtime: f64,
    querytime: f64,
    params: &str,
) -> hdf5::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let file = File::create(path)?;

    let knns_ds = file
        .new_dataset::<i64>()
        .shape((n_queries, k))
        .create("knns")?;
    knns_ds.write_raw(knns)?;

    let dists_ds = file
        .new_dataset::<f32>()
        .shape((n_queries, k))
        .create("dists")?;
    dists_ds.write_raw(dists)?;

    let algo_v = VarLenUnicode::from_str(algo).unwrap();
    let task_v = VarLenUnicode::from_str(task).unwrap();
    let params_v = VarLenUnicode::from_str(params).unwrap();

    file.new_attr::<VarLenUnicode>()
        .create("algo")?
        .write_scalar(&algo_v)?;
    file.new_attr::<VarLenUnicode>()
        .create("task")?
        .write_scalar(&task_v)?;
    file.new_attr::<VarLenUnicode>()
        .create("params")?
        .write_scalar(&params_v)?;
    file.new_attr::<f64>()
        .create("buildtime")?
        .write_scalar(&buildtime)?;
    file.new_attr::<f64>()
        .create("querytime")?
        .write_scalar(&querytime)?;

    Ok(())
}
