use std::f64::consts::PI;

use crate::car::Car;
use crate::centerline::{Centerline, Frenet};
use crate::map::OccGrid;

pub struct Obs {
    pub scans: Vec<Vec<f64>>,
    pub poses: Vec<[f64; 3]>,
    pub cols: Vec<bool>,
    pub rewards: Vec<f64>,
}

pub struct Sim {
    pub map: OccGrid,
    pub cars: Vec<Car>,
    pub centerline: Centerline,
    pub frenet: Vec<Frenet>,
    prev_s: Vec<f64>,
    pub dt: f64,
    pub n_beams: usize,
    pub fov: f64,
    pub max_range: f64,
}

impl Sim {
    pub fn new(yaml: &str, n: usize) -> Self {
        let map = OccGrid::load(yaml);
        let centerline = Centerline::from_map(&map);
        Self {
            map,
            centerline,
            cars: vec![
                Car {
                    x: 0.0,
                    y: 0.0,
                    theta: 0.0,
                    velocity: 0.0,
                    steering: 0.0
                };
                n
            ],
            frenet: vec![Frenet::default(); n],
            prev_s: vec![0.0; n],
            dt: 0.01,
            n_beams: 1081,
            fov: 270.0 * PI / 180.0,
            max_range: 30.0,
        }
    }
    pub fn reset(&mut self, poses: &[[f64; 3]]) -> Obs {
        for (i, (c, p)) in self.cars.iter_mut().zip(poses).enumerate() {
            *c = Car {
                x: p[0],
                y: p[1],
                theta: p[2],
                velocity: 0.0,
                steering: 0.0,
            };
            self.frenet[i] = self.centerline.frenet(p[0], p[1], p[2]);
            self.prev_s[i] = self.frenet[i].s;
        }
        self.observe()
    }
    pub fn reset_single(&mut self, pose: &[f64; 3], i: usize) {
        self.cars[i] = Car {
            x: pose[0],
            y: pose[1],
            theta: pose[2],
            velocity: 0.0,
            steering: 0.0,
        };
        self.frenet[i] = self.centerline.frenet(pose[0], pose[1], pose[2]);
        self.prev_s[i] = self.frenet[i].s;
    }
    pub fn step(&mut self, actions: &[[f64; 2]]) -> Obs {
        for (c, a) in self.cars.iter_mut().zip(actions) {
            c.step(a[0], a[1], self.dt);
        }
        for i in 0..self.cars.len() {
            self.frenet[i] =
                self.centerline
                    .frenet(self.cars[i].x, self.cars[i].y, self.cars[i].theta);
        }
        let mut obs = self.observe();
        let half = self.centerline.total / 2.0;
        obs.rewards = (0..self.cars.len())
            .map(|i| {
                if obs.cols[i] {
                    return -1.0;
                }
                let mut ds = self.frenet[i].s - self.prev_s[i];
                if ds > half {
                    ds -= self.centerline.total;
                }
                if ds < -half {
                    ds += self.centerline.total;
                }
                ds
            })
            .collect();
        for i in 0..self.cars.len() {
            self.prev_s[i] = self.frenet[i].s;
        }
        obs
    }
    pub fn observe(&self) -> Obs {
        let (nb, fov, mr) = (self.n_beams, self.fov, self.max_range);
        let scans = self
            .cars
            .iter()
            .map(|c| {
                (0..nb)
                    .map(|i| {
                        self.map.raycast(
                            c.x,
                            c.y,
                            c.theta - fov / 2.0 + fov * i as f64 / (nb - 1) as f64,
                            mr,
                        )
                    })
                    .collect()
            })
            .collect();
        let poses = self.cars.iter().map(|c| [c.x, c.y, c.theta]).collect();
        let cols = self
            .cars
            .iter()
            .map(|c| self.map.occupied(c.x, c.y))
            .collect();
        Obs {
            scans,
            poses,
            cols,
            rewards: vec![0.0; self.cars.len()],
        }
    }
}
