use crate::sim::{Obs, Sim};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
pub struct F1TenthEnv {
    sim: Sim,
    n_agents: usize,
    max_steps: usize,
    current_step: usize,
    prev_poses: Vec<[f64; 3]>,
    initial_poses: Vec<[f64; 3]>,
}

#[pymethods]
impl F1TenthEnv {
    #[new]
    #[pyo3(signature = (map_path, n_agents=1, max_steps=1000))]
    fn new(map_path: String, n_agents: usize, max_steps: usize) -> PyResult<Self> {
        Ok(Self {
            sim: Sim::new(&map_path, n_agents),
            n_agents,
            max_steps,
            current_step: 0,
            prev_poses: vec![[0.0; 3]; n_agents],
            initial_poses: vec![[0.0; 3]; n_agents],
        })
    }

    /// Reset the environment
    ///
    /// Args:
    ///     poses: Optional list of [x, y, theta] poses for each agent.
    ///            If None, uses [0, 0, 0] for all agents.
    ///     seed: Optional random seed (not used currently)
    ///
    /// Returns:
    ///     observation: Dictionary with 'scans' and 'poses'
    ///     info: Dictionary with additional information
    #[pyo3(signature = (poses=None, seed=None))]
    fn reset(
        &mut self,
        py: Python<'_>,
        poses: Option<Vec<Vec<f64>>>,
        seed: Option<u64>,
    ) -> PyResult<(PyObject, PyObject)> {
        self.current_step = 0;

        let poses_data = if let Some(p) = poses {
            p.iter().map(|v| [v[0], v[1], v[2]]).collect::<Vec<_>>()
        } else {
            vec![[0.0, 0.0, 0.0]; self.n_agents]
        };

        self.initial_poses = poses_data.clone();
        self.prev_poses = poses_data.clone();

        let obs = self.sim.reset(&poses_data);

        let obs_dict = Self::obs_to_dict(py, &obs)?;
        let info = PyDict::new(py);
        info.set_item("step", 0)?;

        Ok((obs_dict.into(), info.into()))
    }

    /// Step the environment
    ///
    /// Args:
    ///     actions: Array of shape (n_agents, 2) where each row is [steering, speed]
    ///
    /// Returns:
    ///     observation: Dictionary with 'scans' and 'poses'
    ///     reward: Array of rewards for each agent
    ///     terminated: Array of booleans indicating if each agent crashed
    ///     truncated: Array of booleans indicating if max steps reached
    ///     info: Dictionary with additional information
    fn step(
        &mut self,
        py: Python<'_>,
        actions: &PyArray2<f64>,
    ) -> PyResult<(PyObject, PyObject, PyObject, PyObject, PyObject)> {
        // Convert actions to Vec<[f64; 2]>
        let actions_vec: Vec<[f64; 2]> = actions
            .readonly()
            .as_array()
            .outer_iter()
            .map(|row| [row[0], row[1]])
            .collect();

        // Step simulation
        let obs = self.sim.step(&actions_vec);
        self.current_step += 1;

        // Calculate rewards
        let rewards = self.calculate_rewards(&obs);

        // Determine termination
        let terminated: Vec<bool> = obs.cols.clone();
        let truncated = vec![self.current_step >= self.max_steps; self.n_agents];

        // Reset crashed agents
        for (i, &crashed) in obs.cols.iter().enumerate() {
            if crashed {
                self.sim.reset_single(&self.initial_poses[i], i);
            }
        }

        // Update previous poses
        self.prev_poses = obs.poses.clone();

        // Build return values
        let obs_dict = Self::obs_to_dict(py, &obs)?;
        let reward_arr = rewards.to_pyarray(py);
        let terminated_arr = terminated.to_pyarray(py);
        let truncated_arr = truncated.to_pyarray(py);

        let info = PyDict::new(py);
        info.set_item("step", self.current_step)?;
        info.set_item("collisions", obs.cols.clone())?;

        Ok((
            obs_dict.into(),
            reward_arr.into(),
            terminated_arr.into(),
            truncated_arr.into(),
            info.into(),
        ))
    }

    /// Get observation space information
    fn observation_space(&self, py: Python<'_>) -> PyResult<PyObject> {
        let space = PyDict::new(py);
        space.set_item("scans_shape", (self.n_agents, self.sim.n_beams))?;
        space.set_item("poses_shape", (self.n_agents, 3))?;
        space.set_item("scan_range", (0.0, self.sim.max_range))?;
        Ok(space.into())
    }

    /// Get action space information
    fn action_space(&self, py: Python<'_>) -> PyResult<PyObject> {
        let space = PyDict::new(py);
        space.set_item("shape", (self.n_agents, 2))?;
        space.set_item("steering_range", (-0.4189, 0.4189))?;
        space.set_item("speed_range", (-5.0, 20.0))?;
        Ok(space.into())
    }

    /// Get current observation without stepping
    fn get_observation(&self, py: Python<'_>) -> PyResult<PyObject> {
        let obs = self.sim.observe();
        let obs_dict = Self::obs_to_dict(py, &obs)?;
        Ok(obs_dict.into())
    }

    /// Get number of agents
    #[getter]
    fn n_agents(&self) -> usize {
        self.n_agents
    }

    /// Get current step count
    #[getter]
    fn current_step(&self) -> usize {
        self.current_step
    }

    /// Render the environment (placeholder)
    #[pyo3(signature = (mode="human"))]
    fn render(&self, mode: &str) -> PyResult<()> {
        // Placeholder for rendering functionality
        Ok(())
    }

    /// Close the environment
    fn close(&self) -> PyResult<()> {
        Ok(())
    }
}

impl F1TenthEnv {
    /// Convert Obs struct to Python dictionary
    fn obs_to_dict<'py>(py: Python<'py>, obs: &Obs) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py);

        // Convert scans to 2D numpy array
        let n_agents = obs.scans.len();
        let n_beams = obs.scans[0].len();
        let mut scans_flat = Vec::with_capacity(n_agents * n_beams);
        for scan in &obs.scans {
            scans_flat.extend(scan);
        }
        let scans_arr = PyArray2::from_vec2(py, &obs.scans)?;
        dict.set_item("scans", scans_arr)?;

        // Convert poses to 2D numpy array
        let poses_arr = PyArray2::from_vec2(
            py,
            &obs.poses.iter().map(|p| p.to_vec()).collect::<Vec<_>>(),
        )?;
        dict.set_item("poses", poses_arr)?;

        Ok(dict)
    }

    /// Calculate rewards based on progress and collisions
    fn calculate_rewards(&self, obs: &Obs) -> Vec<f64> {
        obs.poses
            .iter()
            .zip(&self.prev_poses)
            .zip(&obs.cols)
            .map(|((curr, prev), &crashed)| {
                if crashed {
                    -100.0 // Large penalty for crashing
                } else {
                    // Reward for forward progress
                    let dx = curr[0] - prev[0];
                    let dy = curr[1] - prev[1];
                    let progress = (dx * curr[2].cos() + dy * curr[2].sin()).abs();

                    // Reward for speed (encourage faster movement)
                    let speed_reward = self
                        .sim
                        .cars
                        .iter()
                        .enumerate()
                        .find(|(i, _)| obs.poses[*i] == *curr)
                        .map(|(_, c)| c.velocity.abs() * 0.1)
                        .unwrap_or(0.0);

                    progress + speed_reward - 0.01 // Small penalty per step
                }
            })
            .collect()
    }
}

/// Python module initialization
#[pymodule]
fn f1tenth_gym(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<F1TenthEnv>()?;
    Ok(())
}
