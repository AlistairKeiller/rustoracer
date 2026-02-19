use image::GrayImage;
use imageproc::distance_transform::euclidean_squared_distance_transform;
use serde::Deserialize;

#[derive(Deserialize)]
struct MapMeta {
    image: String,
    resolution: f64,
    origin: [f64; 3],
}

pub struct OccGrid {
    pub img: GrayImage,
    pub occupied: Vec<bool>,
    pub edt: Vec<f64>,
    pub res: f64,
    pub ox: f64,
    pub oy: f64,
}

impl OccGrid {
    pub fn load(yaml: &str) -> Self {
        let m: MapMeta = serde_saphyr::from_str(&std::fs::read_to_string(yaml).unwrap()).unwrap();
        let dir = std::path::Path::new(yaml).parent().unwrap();
        let img = image::open(dir.join(&m.image)).unwrap().into_luma8();
        let occupied = img.pixels().map(|p| p.0[0] < 128).collect::<Vec<bool>>();
        let mut occupied_image = img.clone();
        for pixel in occupied_image.pixels_mut() {
            pixel.0[0] = if pixel.0[0] < 128 { 255 } else { 0 };
        }
        let edt = euclidean_squared_distance_transform(&occupied_image);
        Self {
            img,
            occupied,
            edt: edt
                .pixels()
                .map(|p| p.0[0].sqrt() * m.resolution)
                .collect::<Vec<f64>>(),
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
            || self.occupied[px as usize + py as usize * self.img.width() as usize]
    }
    pub fn distance(&self, wx: f64, wy: f64) -> f64 {
        let px = ((wx - self.ox) / self.res) as i32;
        let py = self.img.height() as i32 - 1 - ((wy - self.oy) / self.res) as i32;
        if px < 0 || py < 0 || px >= self.img.width() as i32 || py >= self.img.height() as i32 {
            return 0.0;
        }
        self.edt[px as usize + py as usize * self.img.width() as usize]
    }
    pub fn raycast(&self, x: f64, y: f64, ang: f64, max: f64) -> f64 {
        let (dx, dy) = (ang.cos(), ang.sin());
        let mut t = 0.0;
        while t < max {
            let wx = x + t * dx;
            let wy = y + t * dy;
            let d = self.distance(wx, wy);
            if d < self.res {
                return t;
            }
            t += d;
        }
        max
    }
}
