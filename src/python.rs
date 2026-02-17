use numpy::PyArray1;
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

        fn reset<'py>(
            &mut self,
            py: Python<'py>,
            pose: [f64; 3],
        ) -> (Bound<'py, PyArray1<f64>>, [f64; 3], bool) {
            let o = self.sim.reset(&[pose]);
            (
                PyArray1::from_vec(py, o.scans[0].clone()),
                o.poses[0],
                o.cols[0],
            )
        }

        fn step<'py>(
            &mut self,
            py: Python<'py>,
            steer: f64,
            speed: f64,
        ) -> (Bound<'py, PyArray1<f64>>, [f64; 3], bool) {
            let o = self.sim.step(&[[steer, speed]]);
            (
                PyArray1::from_vec(py, o.scans[0].clone()),
                o.poses[0],
                o.cols[0],
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
    }
}
