use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
fn plan_attention(
    _batch: usize,
    _heads: usize,
    _q_len: usize,
    head_dim: usize,
    dtype: &str,
    causal: bool,
) -> PyResult<PyObject> {
    let py = unsafe { Python::assume_gil_acquired() };
    let out = PyDict::new_bound(py);

    // MVP heuristic only. This is where autotuning will eventually live.
    let supported = (dtype == "fp16" || dtype == "bf16") && head_dim <= 128 && head_dim % 8 == 0;

    if supported {
        out.set_item("backend", "cuda_flash_attention")?;
        let config = PyDict::new_bound(py);
        config.set_item("head_dim", head_dim)?;
        config.set_item("causal", causal)?;
        config.set_item("q_tile", 64)?;
        config.set_item("k_tile", 64)?;
        config.set_item("num_warps", 4)?;
        out.set_item("config", config)?;
    } else {
        out.set_item("backend", "torch_sdp")?;
        out.set_item("config", PyDict::new_bound(py))?;
    }

    Ok(out.into_py(py))
}

#[pymodule]
fn _planner_ext(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(plan_attention, m)?)?;
    Ok(())
}

