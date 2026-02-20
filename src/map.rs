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
    w: i32,
    h: i32,
    wu: usize,
    pub img: GrayImage,
    pub edt: Vec<f64>,
    pub skeleton: GrayImage,
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
        Self {
            inv_res: 1.0 / m.resolution,
            w: img.width() as i32,
            h: img.height() as i32,
            wu: img.width() as usize,
            img,
            edt: edt.pixels().map(|p| p.0[0].sqrt() * m.resolution).collect(),
            skeleton,
            res: m.resolution,
            ox: m.origin[0],
            oy: m.origin[1],
        }
    }
    pub fn ordered_skeleton(&self, start_x: f64, start_y: f64) -> Vec<[f64; 2]> {
        let mut pts: Vec<[f64; 2]> = self
            .skeleton
            .enumerate_pixels()
            .filter(|(_, _, p)| p.0[0] != 255)
            .map(|(px, py, _)| {
                [
                    px as f64 * self.res + self.ox,
                    (self.skeleton.height() as f64 - 1.0 - py as f64) * self.res + self.oy,
                ]
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
                let da = (a[0] - start_x).powi(2) + (a[1] - start_y).powi(2);
                let db = (b[0] - start_x).powi(2) + (b[1] - start_y).powi(2);
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
    pub fn distance(&self, wx: f64, wy: f64) -> f64 {
        let px = ((wx - self.ox) * self.inv_res) as i32;
        let py = self.h - 1 - ((wy - self.oy) * self.inv_res) as i32;
        if px < 0 || py < 0 || px >= self.w || py >= self.h {
            return 0.0;
        }
        unsafe { *self.edt.get_unchecked(px as usize + py as usize * self.wu) }
    }
    #[inline]
    pub fn raycast(&self, x: f64, y: f64, ang: f64, max: f64) -> f64 {
        let (dy, dx) = ang.sin_cos();
        let mut t = 0.0;
        while t < max {
            let d = self.distance(x + t * dx, y + t * dy);
            if d < self.res {
                return t;
            }
            t += d;
        }
        max
    }
}
