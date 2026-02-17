use pyo3::prelude::*;

#[pymodule]
mod rustoracer {
    use pyo3::prelude::*;

    use crate::{Obs, Sim};

    #[pyclass]
    struct PySim {
        sim: Sim,
    }

    fn obs_to_tuple(obs: Obs) -> (Vec<f64>, [f64; 3], bool) {
        (
            obs.scans[0].clone(),
            obs.poses[0].clone(),
            obs.cols[0].clone(),
        )
    }

    #[pymethods]
    impl PySim {
        #[new]
        fn new(yaml: &str) -> Self {
            PySim {
                sim: Sim::new(yaml, 1),
            }
        }

        fn step(&mut self, steer: f64, speed: f64) -> (Vec<f64>, [f64; 3], bool) {
            obs_to_tuple(self.sim.step(&[[steer, speed]]))
        }

        fn reset(&mut self) -> (Vec<f64>, [f64; 3], bool) {
            obs_to_tuple(self.sim.reset(&[[0.0, 0.0, 0.0]]))
        }
    }
}
