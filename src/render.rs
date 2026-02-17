use crate::car::Car;
use crate::map::OccGrid;

pub fn render_rgb(map: &OccGrid, cars: &[Car]) -> (Vec<u8>, u32, u32) {
    let (w, h) = (map.img.width(), map.img.height());
    let mut buf = vec![0u8; (h * w * 3) as usize];

    for (i, p) in map.img.pixels().enumerate() {
        let v = p.0[0];
        let j = i * 3;
        buf[j] = v;
        buf[j + 1] = v;
        buf[j + 2] = v;
    }

    for car in cars {
        let cx = ((car.x - map.ox) / map.res) as i32;
        let cy = h as i32 - 1 - ((car.y - map.oy) / map.res) as i32;
        for dy in -3..=3i32 {
            for dx in -3..=3i32 {
                if dx * dx + dy * dy <= 9 {
                    set_px(&mut buf, w, h, cx + dx, cy + dy, [255, 0, 0]);
                }
            }
        }
        for t in 0..8 {
            set_px(
                &mut buf,
                w,
                h,
                cx + (t as f64 * car.theta.cos()) as i32,
                cy - (t as f64 * car.theta.sin()) as i32,
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
