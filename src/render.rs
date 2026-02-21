use crate::car::{Car, LENGTH, WIDTH};
use crate::map::OccGrid;

const HALF_L: f64 = LENGTH / 2.0;
const HALF_W: f64 = WIDTH / 2.0;

pub fn render_rgb(map: &OccGrid, cars: &[Car]) -> (Vec<u8>, u32, u32) {
    let (w, h) = (map.img.width(), map.img.height());
    let mut buf = vec![0u8; (h * w * 3) as usize];
    for (i, p) in map.img.pixels().enumerate() {
        buf[i * 3..][..3].fill(p.0[0]);
    }
    for car in cars {
        let s = 1.0 / map.res;
        let cx = (car.x - map.ox) * s;
        let cy = h as f64 - 1.0 - (car.y - map.oy) * s;
        let (sa, ca) = car.theta.sin_cos();
        let (hl, hw) = (HALF_L * s, HALF_W * s);

        let r = hl.hypot(hw).ceil() as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                let (fx, fy) = (dx as f64, dy as f64);
                if (fx * ca - fy * sa).abs() <= hl && (fx * sa + fy * ca).abs() <= hw {
                    set_px(&mut buf, w, h, cx as i32 + dx, cy as i32 + dy, [255, 0, 0]);
                }
            }
        }

        for t in 0..=(hl as i32 + 2) {
            let ft = t as f64;
            set_px(
                &mut buf,
                w,
                h,
                cx as i32 + (ft * ca) as i32,
                cy as i32 - (ft * sa) as i32,
                [0, 255, 0],
            );
        }
    }
    (buf, h, w)
}

fn set_px(buf: &mut [u8], w: u32, h: u32, x: i32, y: i32, c: [u8; 3]) {
    if (0..w as i32).contains(&x) && (0..h as i32).contains(&y) {
        let i = (y as usize * w as usize + x as usize) * 3;
        buf[i..i + 3].copy_from_slice(&c);
    }
}
