use numpy::{IntoPyArray, PyArray1, PyArray3};
use pyo3::prelude::*;

use crate::Sim;

#[pymodule]
mod rustoracer {
    use numpy::PyReadonlyArray1;

    use crate::car::{STEER_MAX, STEER_MIN, V_MAX, V_MIN};

    use super::*;

    #[pyclass]
    struct PySim {
        sim: Sim,
    }

    #[pymethods]
    impl PySim {
        #[new]
        fn new(yaml: &str, n: usize, max_steps: u32) -> Self {
            Self {
                sim: Sim::new(yaml, n, max_steps),
            }
        }

        fn seed(&mut self, seed: u64) {
            self.sim.seed(seed);
        }

        fn reset<'py>(
            &mut self,
            py: Python<'py>,
        ) -> (
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyArray1<f64>>,
        ) {
            let o = self.sim.reset_zeros();
            (
                o.scans.into_pyarray(py),
                o.rewards.into_pyarray(py),
                o.terminated.into_pyarray(py),
                o.truncated.into_pyarray(py),
                o.state.into_pyarray(py),
            )
        }

        fn step<'py>(
            &mut self,
            py: Python<'py>,
            actions: PyReadonlyArray1<f64>,
        ) -> (
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyArray1<f64>>,
        ) {
            let raw = actions.as_slice().unwrap();
            let rescaled: Vec<f64> = raw
                .chunks(2)
                .flat_map(|a| {
                    [
                        STEER_MIN + (a[0] + 1.0) * 0.5 * (STEER_MAX - STEER_MIN),
                        V_MIN + (a[1] + 1.0) * 0.5 * (V_MAX - V_MIN),
                    ]
                })
                .collect();
            let o = self.sim.step(&rescaled);
            (
                o.scans.into_pyarray(py),
                o.rewards.into_pyarray(py),
                o.terminated.into_pyarray(py),
                o.truncated.into_pyarray(py),
                o.state.into_pyarray(py),
            )
        }

        #[getter]
        fn n_beams(&self) -> usize {
            self.sim.n_beams
        }

        #[getter]
        fn max_range(&self) -> f64 {
            self.sim.max_range
        }

        #[getter]
        fn skeleton<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            let flat: Vec<f64> = self
                .sim
                .map
                .ordered_skeleton
                .iter()
                .flat_map(|p| p.iter().copied())
                .collect();
            flat.into_pyarray(py)
        }

        fn render<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u8>> {
            let (buf, h, w) = crate::render::render_rgb(&self.sim.map, &self.sim.cars);
            numpy::ndarray::Array3::from_shape_vec((h as usize, w as usize, 3), buf)
                .unwrap()
                .into_pyarray(py)
        }

        fn reset_single(&mut self, i: usize) {
            self.sim.reset_single(&[0.0, 0.0, 0.0], i);
        }

        fn observe<'py>(
            &mut self,
            py: Python<'py>,
        ) -> (
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyArray1<f64>>,
        ) {
            let o = self.sim.observe();
            (
                o.scans.into_pyarray(py),
                o.rewards.into_pyarray(py),
                o.terminated.into_pyarray(py),
                o.truncated.into_pyarray(py),
                o.state.into_pyarray(py),
            )
        }
    }
}
