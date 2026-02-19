use crate::map::OccGrid;
use image::{GrayImage, Luma};
use imageproc::distance_transform::euclidean_squared_distance_transform;
use std::f64::consts::PI;

pub struct Centerline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    ss: Vec<f64>,
    pub total: f64,
}

#[derive(Clone, Copy, Default)]
pub struct Frenet {
    pub s: f64,
    pub d: f64,
    pub phi: f64,
}

impl Centerline {
    pub fn from_map(map: &OccGrid) -> Self {
        let (w, h) = map.img.dimensions();
        // Walls = foreground (255), free = background (0)
        let bin = GrayImage::from_fn(w, h, |x, y| {
            Luma([if map.img.get_pixel(x, y).0[0] < 128 {
                255u8
            } else {
                0
            }])
        });
        let edt = euclidean_squared_distance_transform(&bin);

        // Ridge: local max of squared-EDT in either cardinal direction pair
        let px = |x: i32, y: i32| -> f64 {
            if (0..w as i32).contains(&x) && (0..h as i32).contains(&y) {
                edt.get_pixel(x as u32, y as u32).0[0]
            } else {
                0.0
            }
        };
        let mut skel: Vec<(u32, u32)> = Vec::new();
        for y in 1..h as i32 - 1 {
            for x in 1..w as i32 - 1 {
                let v = px(x, y);
                if v <= 1.0 {
                    continue;
                }
                if (v > px(x - 1, y) && v > px(x + 1, y)) || (v > px(x, y - 1) && v > px(x, y + 1))
                {
                    skel.push((x as u32, y as u32));
                }
            }
        }
        if skel.len() < 2 {
            return Self {
                xs: vec![],
                ys: vec![],
                ss: vec![],
                total: 0.0,
            };
        }

        // Order by nearest-neighbor traversal
        let mut used = vec![false; skel.len()];
        let mut ordered = vec![skel[0]];
        used[0] = true;
        for _ in 1..skel.len() {
            let (cx, cy) = *ordered.last().unwrap();
            let (bi, bd) = skel
                .iter()
                .enumerate()
                .filter(|(i, _)| !used[*i])
                .map(|(i, &(px, py))| {
                    (
                        i,
                        (px as i64 - cx as i64).pow(2) + (py as i64 - cy as i64).pow(2),
                    )
                })
                .min_by_key(|&(_, d)| d)
                .unwrap();
            if bd > 50 {
                break;
            }
            used[bi] = true;
            ordered.push(skel[bi]);
        }

        // Subsample & convert to world coords
        let step = (ordered.len() / 200).max(1);
        let xs: Vec<f64> = ordered
            .iter()
            .step_by(step)
            .map(|&(px, _)| map.ox + (px as f64 + 0.5) * map.res)
            .collect();
        let ys: Vec<f64> = ordered
            .iter()
            .step_by(step)
            .map(|&(_, py)| map.oy + (h as f64 - 0.5 - py as f64) * map.res)
            .collect();

        // Cumulative arc length (closed loop)
        let n = xs.len();
        let mut ss = vec![0.0; n];
        for i in 1..n {
            ss[i] = ss[i - 1] + ((xs[i] - xs[i - 1]).powi(2) + (ys[i] - ys[i - 1]).powi(2)).sqrt();
        }
        let total = if n > 1 {
            ss[n - 1] + ((xs[0] - xs[n - 1]).powi(2) + (ys[0] - ys[n - 1]).powi(2)).sqrt()
        } else {
            0.0
        };

        Self { xs, ys, ss, total }
    }

    pub fn frenet(&self, x: f64, y: f64, theta: f64) -> Frenet {
        let n = self.xs.len();
        if n < 2 {
            return Frenet::default();
        }
        // Find closest segment
        let (mut bi, mut bt, mut bd) = (0, 0.0, f64::MAX);
        for i in 0..n {
            let j = (i + 1) % n;
            let (dx, dy) = (self.xs[j] - self.xs[i], self.ys[j] - self.ys[i]);
            let l2 = dx * dx + dy * dy;
            if l2 < 1e-12 {
                continue;
            }
            let t = (((x - self.xs[i]) * dx + (y - self.ys[i]) * dy) / l2).clamp(0.0, 1.0);
            let d2 = (x - self.xs[i] - t * dx).powi(2) + (y - self.ys[i] - t * dy).powi(2);
            if d2 < bd {
                bi = i;
                bt = t;
                bd = d2;
            }
        }
        let j = (bi + 1) % n;
        let (dx, dy) = (self.xs[j] - self.xs[bi], self.ys[j] - self.ys[bi]);
        let len = (dx * dx + dy * dy).sqrt();
        let tang = dy.atan2(dx);
        let seg_s = if bi < n - 1 { self.ss[bi] } else { self.ss[bi] };
        let s = (seg_s + bt * len) % self.total;
        let (px, py) = (self.xs[bi] + bt * dx, self.ys[bi] + bt * dy);
        let d = (x - px) * (-dy / len) + (y - py) * (dx / len);
        let mut phi = theta - tang;
        while phi > PI {
            phi -= 2.0 * PI;
        }
        while phi < -PI {
            phi += 2.0 * PI;
        }
        Frenet { s, d, phi }
    }
}
