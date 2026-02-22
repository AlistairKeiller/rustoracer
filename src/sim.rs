use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::f64::consts::PI;

use crate::car::Car;
use crate::map::OccGrid;

pub struct Obs {
    pub scans: Vec<f64>,
    pub rewards: Vec<f64>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub state: Vec<f64>,
}

pub struct Sim {
    pub map: OccGrid,
    pub cars: Vec<Car>,
    pub dt: f64,
    pub n_beams: usize,
    pub fov: f64,
    pub max_range: f64,
    pub rng: SmallRng,
    pub waypoint_idx: Vec<usize>,
    pub steps: Vec<u32>,
    pub max_steps: u32,
}

impl Sim {
    pub fn new(yaml: &str, n: usize, max_steps: u32) -> Self {
        Self {
            map: OccGrid::load(yaml),
            cars: vec![
                Car {
                    x: 0.0,
                    y: 0.0,
                    theta: 0.0,
                    velocity: 0.0,
                    steering: 0.0,
                    yaw_rate: 0.0,
                    slip_angle: 0.0,
                };
                n
            ],
            dt: 0.01,
            n_beams: 1081,
            fov: 270.0 * PI / 180.0,
            max_range: 30.0,
            rng: SmallRng::seed_from_u64(0),
            waypoint_idx: vec![0; n],
            steps: vec![0; n],
            max_steps,
        }
    }
    pub fn seed(&mut self, seed: u64) {
        self.rng = SmallRng::seed_from_u64(seed);
    }
    pub fn reset_zeros(&mut self) -> Obs {
        self.waypoint_idx = vec![0; self.cars.len()];
        self.steps = vec![0; self.cars.len()];
        for c in self.cars.iter_mut() {
            *c = Car {
                x: 0.0,
                y: 0.0,
                theta: 0.0,
                velocity: 0.0,
                steering: 0.0,
                yaw_rate: 0.0,
                slip_angle: 0.0,
            };
        }
        self.observe()
    }
    pub fn reset(&mut self, poses: &[[f64; 3]]) -> Obs {
        self.waypoint_idx = vec![0; self.cars.len()];
        self.steps = vec![0; self.cars.len()];
        for (c, p) in self.cars.iter_mut().zip(poses) {
            *c = Car {
                x: p[0],
                y: p[1],
                theta: p[2],
                velocity: 0.0,
                steering: 0.0,
                yaw_rate: 0.0,
                slip_angle: 0.0,
            };
        }
        self.observe()
    }
    pub fn reset_single(&mut self, pose: &[f64; 3], i: usize) {
        self.waypoint_idx[i] = 0;
        self.steps[i] = 0;
        self.cars[i] = Car {
            x: pose[0],
            y: pose[1],
            theta: pose[2],
            velocity: 0.0,
            steering: 0.0,
            yaw_rate: 0.0,
            slip_angle: 0.0,
        };
    }
    pub fn step(&mut self, actions: &[f64]) -> Obs {
        for (c, a) in self.cars.iter_mut().zip(actions.chunks(2)) {
            c.step(a[0], a[1], self.dt);
        }
        self.observe()
    }
    pub fn observe(&mut self) -> Obs {
        let (nb, fov, mr) = (self.n_beams, self.fov, self.max_range);
        let rng = &mut self.rng;
        let scans: Vec<f64> = self
            .cars
            .iter()
            .map(|c| {
                (0..nb)
                    .into_iter()
                    .map(|i| {
                        self.map.raycast(
                            c.x,
                            c.y,
                            c.theta - fov / 2.0 + fov * i as f64 / (nb - 1) as f64,
                            mr,
                            rng,
                        )
                    })
                    .collect::<Vec<f64>>()
            })
            .flatten()
            .collect();
        let n_wps = self.map.ordered_skeleton.len();
        let rewards: Vec<f64> = self
            .cars
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let prev_idx = self.waypoint_idx[i];
                let nearest = (0..n_wps)
                    .min_by(|&a, &b| {
                        let wp_a = self.map.ordered_skeleton[a];
                        let wp_b = self.map.ordered_skeleton[b];
                        let da = (wp_a[0] - c.x).powi(2) + (wp_a[1] - c.y).powi(2);
                        let db = (wp_b[0] - c.x).powi(2) + (wp_b[1] - c.y).powi(2);
                        da.partial_cmp(&db).unwrap()
                    })
                    .unwrap();
                self.waypoint_idx[i] = nearest;
                let mut delta = nearest as f64 - prev_idx as f64;
                if delta > n_wps as f64 / 2.0 {
                    delta -= n_wps as f64;
                } else if delta < -(n_wps as f64 / 2.0) {
                    delta += n_wps as f64;
                }
                delta
            })
            .collect();
        let terminated: Vec<bool> = self.cars.iter().map(|c| self.map.car_collides(c)).collect();
        let truncated: Vec<bool> = self.steps.iter().map(|&s| s >= self.max_steps).collect();
        for i in 0..self.cars.len() {
            if terminated[i] || truncated[i] {
                self.reset_single(&[0.0, 0.0, 0.0], i);
            }
        }
        let state: Vec<f64> = self
            .cars
            .iter()
            .map(|c| {
                [
                    c.x,
                    c.y,
                    c.theta,
                    c.velocity,
                    c.steering,
                    c.yaw_rate,
                    c.slip_angle,
                ]
            })
            .flatten()
            .collect();
        Obs {
            scans,
            rewards,
            terminated,
            truncated,
            state,
        }
    }
}
