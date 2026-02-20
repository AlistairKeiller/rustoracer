use crate::skeleton::thin_image_edges;
use image::GrayImage;
use imageproc::distance_transform::euclidean_squared_distance_transform;
use serde::Deserialize;
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
    pub res: f64,
    pub ox: f64,
    pub oy: f64,
}

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
        view_image(&occupied_image, "occupied");
        let max_edt = edt.pixels().map(|p| p.0[0]).fold(0.0f64, f64::max);
        let edt_gray = GrayImage::from_fn(edt.width(), edt.height(), |x, y| {
            let val = edt.get_pixel(x, y).0[0];
            image::Luma([(val / max_edt * 255.0) as u8])
        });
        view_image(&edt_gray, "edt");
        view_image(&skeleton, "skeleton");
        Self {
            inv_res: 1.0 / m.resolution,
            w: img.width() as i32,
            h: img.height() as i32,
            wu: img.width() as usize,
            img,
            edt: edt.pixels().map(|p| p.0[0].sqrt() * m.resolution).collect(),
            res: m.resolution,
            ox: m.origin[0],
            oy: m.origin[1],
        }
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
