use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;

use crate::Sim;

#[pymodule]
mod rustoracer {
    use super::*;

    #[pyclass]
    struct PySim {
        sim: Sim,
    }

    #[pymethods]
    impl PySim {
        #[new]
        fn new(yaml: &str) -> Self {
            Self {
                sim: Sim::new(yaml, 1),
            }
        }

        fn seed(&mut self, seed: u64) {
            self.sim.seed(seed);
        }

        fn reset<'py>(
            &mut self,
            py: Python<'py>,
            pose: [f64; 3],
        ) -> (Bound<'py, PyArray1<f64>>, [f64; 7], bool, f64) {
            let o = self.sim.reset(&[pose]);
            (
                PyArray1::from_vec(py, o.scans[0].clone()),
                o.state[0],
                o.cols[0],
                o.progress[0],
            )
        }

        fn step<'py>(
            &mut self,
            py: Python<'py>,
            steer: f64,
            speed: f64,
        ) -> (Bound<'py, PyArray1<f64>>, [f64; 7], bool, f64) {
            let o = self.sim.step(&[[steer, speed]]);
            (
                PyArray1::from_vec(py, o.scans[0].clone()),
                o.state[0],
                o.cols[0],
                o.progress[0],
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
        fn skeleton<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
            let pts = self.sim.map.ordered_skeleton.clone();
            let n = pts.len();
            let flat: Vec<f64> = pts.into_iter().flat_map(|p| p).collect();
            numpy::ndarray::Array2::from_shape_vec((n, 2), flat)
                .unwrap()
                .into_pyarray(py)
        }

        fn render<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u8>> {
            let (buf, h, w) = crate::render::render_rgb(&self.sim.map, &self.sim.cars);
            numpy::ndarray::Array3::from_shape_vec((h as usize, w as usize, 3), buf)
                .unwrap()
                .into_pyarray(py)
        }
    }
}
