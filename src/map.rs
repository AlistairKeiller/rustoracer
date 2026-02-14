use image::GrayImage;
use serde::Deserialize;

#[derive(Deserialize)]
struct MapMeta {
    image: String,
    resolution: f64,
    origin: [f64; 3],
}

pub struct OccGrid {
    img: GrayImage,
    res: f64,
    ox: f64,
    oy: f64,
}

impl OccGrid {
    pub fn load(yaml: &str) -> Self {
        let m: MapMeta = serde_saphyr::from_str(&std::fs::read_to_string(yaml).unwrap()).unwrap();
        let dir = std::path::Path::new(yaml).parent().unwrap();
        let img = image::open(dir.join(&m.image)).unwrap().into_luma8();
        Self {
            img,
            res: m.resolution,
            ox: m.origin[0],
            oy: m.origin[1],
        }
    }
    pub fn occupied(&self, wx: f64, wy: f64) -> bool {
        let px = ((wx - self.ox) / self.res) as i32;
        let py = self.img.height() as i32 - 1 - ((wy - self.oy) / self.res) as i32;
        px < 0
            || py < 0
            || px >= self.img.width() as i32
            || py >= self.img.height() as i32
            || self.img.get_pixel(px as u32, py as u32).0[0] < 128
    }
    pub fn raycast(&self, x: f64, y: f64, ang: f64, max: f64) -> f64 {
        let (dx, dy) = (ang.cos(), ang.sin());
        let step = self.res * 0.5;
        let mut t = 0.0;
        while t < max {
            t += step;
            if self.occupied(x + t * dx, y + t * dy) {
                return t;
            }
        }
        max
    }
}
