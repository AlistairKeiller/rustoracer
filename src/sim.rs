use crate::car::Car;
use crate::map::OccGrid;

pub struct Obs {
    pub scans: Vec<Vec<f64>>,
    pub poses: Vec<[f64; 3]>,
}

pub struct Sim {
    pub map: OccGrid,
    pub cars: Vec<Car>,
    pub dt: f64,
    pub n_beams: usize,
    pub fov: f64,
    pub max_range: f64,
}

impl Sim {
    pub fn new(yaml: &str, n: usize) -> Self {
        Self {
            map: OccGrid::load(yaml),
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
            dt: 0.01,
            n_beams: 1080,
            fov: 4.7,
            max_range: 30.0,
        }
    }
    pub fn reset(&mut self, poses: &[[f64; 3]]) -> Obs {
        for (c, p) in self.cars.iter_mut().zip(poses) {
            *c = Car {
                x: p[0],
                y: p[1],
                theta: p[2],
                velocity: 0.0,
                steering: 0.0,
            };
        }
        self.observe()
    }
    pub fn step(&mut self, actions: &[[f64; 2]]) -> (Obs, Vec<bool>) {
        for (c, a) in self.cars.iter_mut().zip(actions) {
            c.step(a[0], a[1], self.dt);
        }
        let cols = self
            .cars
            .iter()
            .map(|c| self.map.occupied(c.x, c.y))
            .collect();
        (self.observe(), cols)
    }
    pub fn observe(&self) -> Obs {
        let (nb, fov, mr) = (self.n_beams, self.fov, self.max_range);
        let scans: Vec<Vec<f64>> = self
            .cars
            .iter()
            .map(|c| {
                (0..nb)
                    .into_iter()
                    .map(|i| {
                        let ang = c.theta - fov / 2.0 + fov * i as f64 / (nb - 1) as f64;
                        self.map.raycast(c.x, c.y, ang, mr)
                    })
                    .collect()
            })
            .collect();
        let poses = self.cars.iter().map(|c| [c.x, c.y, c.theta]).collect();
        Obs { scans, poses }
    }
}
