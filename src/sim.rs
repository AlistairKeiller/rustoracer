use kiddo::SquaredEuclidean;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::car::Car;
use crate::map::OccGrid;

pub struct Obs<'a> {
    pub scans: &'a [f64],
    pub rewards: &'a [f64],
    pub terminated: &'a [bool],
    pub truncated: &'a [bool],
    pub state: &'a [f64],
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
    beam_sin_cos: Vec<(f64, f64)>,
    buf_terminated: Vec<bool>,
    buf_truncated: Vec<bool>,
    buf_rewards: Vec<f64>,
    buf_scans: Vec<f64>,
    buf_state: Vec<f64>,
}

impl Sim {
    pub fn new(yaml: &str, n: usize, max_steps: u32) -> Self {
        let n_beams: usize = 108;
        let fov: f64 = 270.0 * PI / 180.0;
        let beam_sin_cos: Vec<(f64, f64)> = (0..n_beams)
            .map(|i| -fov / 2.0 + fov * i as f64 / (n_beams - 1) as f64)
            .map(|a| a.sin_cos())
            .collect();
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
            dt: 1.0 / 60.0,
            n_beams,
            fov,
            max_range: 30.0,
            rng: SmallRng::seed_from_u64(0),
            waypoint_idx: vec![0; n],
            steps: vec![0; n],
            max_steps,
            beam_sin_cos,
            buf_terminated: vec![false; n],
            buf_truncated: vec![false; n],
            buf_rewards: vec![0.0; n],
            buf_scans: vec![0.0; n * (n_beams + 2)],
            buf_state: vec![0.0; n * 7],
        }
    }

    pub fn seed(&mut self, seed: u64) {
        self.rng = SmallRng::seed_from_u64(seed);
    }

    fn nearest_waypoint(&self, x: f64, y: f64) -> usize {
        let n_wps = self.map.ordered_skeleton.len();
        (0..n_wps)
            .min_by(|&a, &b| {
                let wa = self.map.ordered_skeleton[a];
                let wb = self.map.ordered_skeleton[b];
                let da = (wa[0] - x).powi(2) + (wa[1] - y).powi(2);
                let db = (wb[0] - x).powi(2) + (wb[1] - y).powi(2);
                da.total_cmp(&db)
            })
            .unwrap_or(0)
    }

    pub fn reset_zeros(&mut self) -> Obs {
        self.steps.fill(0);
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
        let nearest = self.nearest_waypoint(0.0, 0.0);
        self.waypoint_idx.fill(nearest);
        self.observe()
    }

    pub fn reset(&mut self, poses: &[[f64; 3]]) -> Obs<'_> {
        self.steps.fill(0);
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
        for i in 0..self.cars.len() {
            self.waypoint_idx[i] = self.nearest_waypoint(self.cars[i].x, self.cars[i].y);
        }
        self.observe()
    }

    pub fn reset_single(&mut self, pose: &[f64; 3], i: usize) {
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
        self.waypoint_idx[i] = self.nearest_waypoint(pose[0], pose[1]);
    }

    pub fn step(&mut self, actions: &[f64]) -> Obs {
        for (c, a) in self.cars.iter_mut().zip(actions.chunks(2)) {
            c.step(a[0], a[1], self.dt);
        }
        for s in self.steps.iter_mut() {
            *s += 1;
        }
        self.observe()
    }

    pub fn observe(&mut self) -> Obs<'_> {
        let n = self.cars.len();
        let n_wps = self.map.ordered_skeleton.len();
        let n_beams = self.n_beams;
        let max_range = self.max_range;

        let random_resets: Vec<[f64; 3]> = (0..n)
            .map(|_| {
                let rand_idx = self.rng.random_range(0..n_wps);
                let wp = self.map.ordered_skeleton[rand_idx];
                let next_wp = self.map.ordered_skeleton[(rand_idx + 1) % n_wps];
                let theta = (next_wp[1] - wp[1]).atan2(next_wp[0] - wp[0]);
                [wp[0], wp[1], theta]
            })
            .collect();

        let map = &self.map;
        let beam_sin_cos = &self.beam_sin_cos;

        self.buf_terminated
            .par_iter_mut()
            .zip(self.buf_truncated.par_iter_mut())
            .zip(self.buf_rewards.par_iter_mut())
            .zip(self.cars.par_iter_mut())
            .zip(self.waypoint_idx.par_iter_mut())
            .zip(self.steps.par_iter_mut())
            .zip(self.buf_scans.par_chunks_mut(n_beams + 2))
            .zip(self.buf_state.par_chunks_mut(7))
            .zip(random_resets.par_iter())
            .for_each(
                |(
                    (((((((terminated, truncated), reward), car), wp_idx), step), scan), state),
                    reset,
                )| {
                    *terminated = map.car_collides(car);
                    *truncated = *step >= self.max_steps;

                    let prev_idx = *wp_idx;
                    let nearest = self
                        .map
                        .skeleton_tree
                        .nearest_one::<SquaredEuclidean>(&[car.x, car.y])
                        .item;
                    *wp_idx = nearest;

                    let mut delta = nearest as f64 - prev_idx as f64;
                    if delta > n_wps as f64 / 2.0 {
                        delta -= n_wps as f64;
                    } else if delta < -(n_wps as f64 / 2.0) {
                        delta += n_wps as f64;
                    }
                    *reward = delta / n_wps as f64 * 100.0 - if *terminated { 100.0 } else { 0.0 };

                    if *terminated || *truncated {
                        *step = 0;
                        *car = Car {
                            x: reset[0],
                            y: reset[1],
                            theta: reset[2],
                            velocity: 0.0,
                            steering: 0.0,
                            yaw_rate: 0.0,
                            slip_angle: 0.0,
                        };
                        *wp_idx = {
                            let (cx, cy) = (car.x, car.y);
                            (0..n_wps)
                                .min_by(|&a, &b| {
                                    let wa = map.ordered_skeleton[a];
                                    let wb = map.ordered_skeleton[b];
                                    let da = (wa[0] - cx).powi(2) + (wa[1] - cy).powi(2);
                                    let db = (wb[0] - cx).powi(2) + (wb[1] - cy).powi(2);
                                    da.total_cmp(&db)
                                })
                                .unwrap_or(0)
                        };
                    }

                    let (sin_h, cos_h) = car.theta.sin_cos();
                    for (j, &(sin_a, cos_a)) in beam_sin_cos.iter().enumerate() {
                        let dx = cos_h * cos_a - sin_h * sin_a;
                        let dy = sin_h * cos_a + cos_h * sin_a;
                        scan[j] = map.raycast(car.x, car.y, dx, dy, max_range);
                    }
                    scan[n_beams] = car.velocity;
                    scan[n_beams + 1] = car.steering;

                    state[0] = car.x;
                    state[1] = car.y;
                    state[2] = car.theta;
                    state[3] = car.velocity;
                    state[4] = car.steering;
                    state[5] = car.yaw_rate;
                    state[6] = car.slip_angle;
                },
            );

        Obs {
            scans: &self.buf_scans,
            rewards: &self.buf_rewards,
            terminated: &self.buf_terminated,
            truncated: &self.buf_truncated,
            state: &self.buf_state,
        }
    }
}
