use crate::car::{Car, LENGTH, WIDTH};
use crate::skeleton::thin_image_edges;
use image::GrayImage;
use imageproc::distance_transform::euclidean_squared_distance_transform;
use serde::Deserialize;
#[cfg(feature = "show_images")]
use show_image::{ImageInfo, ImageView, create_window};

#[derive(Deserialize)]
struct MapMeta {
    image: String,
    resolution: f64,
    origin: [f64; 3],
}

pub struct OccGrid {
    inv_res: f64,
    pub img: GrayImage,
    pub edt: Vec<f64>,
    pub skeleton: GrayImage,
    pub ordered_skeleton: Vec<[f64; 2]>,
    pub res: f64,
    pub ox: f64,
    pub oy: f64,
}

#[cfg(feature = "show_images")]
fn view_image(img: &GrayImage, title: &str) {
    let window = create_window(title, Default::default()).unwrap();
    let image_view = ImageView::new(ImageInfo::mono8(img.width(), img.height()), img.as_raw());
    window.set_image(title, image_view).unwrap();
}

impl OccGrid {
    pub fn load(yaml: &str) -> Self {
        let m: MapMeta = serde_saphyr::from_str(&std::fs::read_to_string(yaml).unwrap()).unwrap();
        let dir = std::path::Path::new(yaml).parent().unwrap();
        let img = image::open(dir.join(&m.image)).unwrap().into_luma8();
        let mut occupied_image = img.clone();
        for pixel in occupied_image.pixels_mut() {
            pixel.0[0] = if pixel.0[0] < 128 { 255 } else { 0 };
        }
        let edt = euclidean_squared_distance_transform(&occupied_image);
        let skeleton = thin_image_edges(&occupied_image);
        #[cfg(feature = "show_images")]
        view_image(&occupied_image, "occupied");
        #[cfg(feature = "show_images")]
        view_image(&skeleton, "skeleton");
        let mut map = Self {
            inv_res: 1.0 / m.resolution,
            img,
            edt: edt.pixels().map(|p| p.0[0].sqrt() * m.resolution).collect(),
            skeleton,
            ordered_skeleton: Vec::new(),
            res: m.resolution,
            ox: m.origin[0],
            oy: m.origin[1],
        };
        map.ordered_skeleton = map.ordered_skeleton();
        map
    }
    pub fn ordered_skeleton(&self) -> Vec<[f64; 2]> {
        let mut pts: Vec<[f64; 2]> = self
            .skeleton
            .enumerate_pixels()
            .filter(|(_, _, p)| p.0[0] != 255)
            .map(|(px, py, _)| {
                let (x, y) = self.pixels_to_position(px, py);
                [x, y]
            })
            .collect();
        if pts.is_empty() {
            return pts;
        }
        let mut ordered = Vec::with_capacity(pts.len());
        let first = pts
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = a[0].powi(2) + a[1].powi(2);
                let db = b[0].powi(2) + b[1].powi(2);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
            .0;
        ordered.push(pts.swap_remove(first));
        while !pts.is_empty() {
            let last = *ordered.last().unwrap();
            let nearest = pts
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (a[0] - last[0]).powi(2) + (a[1] - last[1]).powi(2);
                    let db = (b[0] - last[0]).powi(2) + (b[1] - last[1]).powi(2);
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap()
                .0;
            ordered.push(pts.swap_remove(nearest));
        }
        ordered
    }
    #[inline]
    pub fn pixels_to_position(&self, px: u32, py: u32) -> (f64, f64) {
        let wx = px as f64 * self.res + self.ox;
        let wy = (self.img.height() - 1 - py) as f64 * self.res + self.oy;
        (wx, wy)
    }
    #[inline]
    pub fn position_to_pixels(&self, wx: f64, wy: f64) -> (u32, u32) {
        let px = ((wx - self.ox) * self.inv_res) as u32;
        let py = self.img.height() - 1 - ((wy - self.oy) * self.inv_res) as u32;
        (px, py)
    }
    #[inline]
    pub fn edt(&self, px: u32, py: u32) -> f64 {
        if (0..self.img.width()).contains(&px) && (0..self.img.height()).contains(&py) {
            unsafe {
                *self
                    .edt
                    .get_unchecked((px + py * self.img.width()) as usize)
            }
        } else {
            0.0
        }
    }
    #[inline]
    pub fn raycast(&self, x: f64, y: f64, ang: f64, max: f64) -> f64 {
        let (dy, dx) = ang.sin_cos();
        let mut t = 0.0;
        while t < max {
            let (px, py) = self.position_to_pixels(x + t * dx, y + t * dy);
            let d = self.edt(px, py);
            if d < self.res {
                return t;
            }
            t += d;
        }
        max
    }
    pub fn car_pixels(&self, car: &Car) -> Vec<(u32, u32)> {
        let (sa, ca) = car.theta.sin_cos();
        let (hl, hw) = (LENGTH / 2.0 * self.inv_res, WIDTH / 2.0 * self.inv_res);
        let (cx, cy) = self.position_to_pixels(car.x, car.y);
        let r = hl.hypot(hw).ceil() as i32;
        let mut out = Vec::new();
        for dy in -r..=r {
            for dx in -r..=r {
                let (fx, fy) = (dx as f64, dy as f64);
                if (fx * ca - fy * sa).abs() <= hl && (fx * sa + fy * ca).abs() <= hw {
                    out.push(((cx as i32 + dx) as u32, (cy as i32 + dy) as u32));
                }
            }
        }
        out
    }

    pub fn car_collides(&self, car: &Car) -> bool {
        self.car_pixels(car)
            .into_iter()
            .any(|(x, y)| self.edt(x, y) < self.res)
    }
}
