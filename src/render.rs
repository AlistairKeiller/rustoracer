use crate::car::Car;
use crate::map::OccGrid;

pub fn render_rgb(map: &OccGrid, cars: &[Car]) -> (Vec<u8>, u32, u32) {
    let (w, h) = (map.img.width(), map.img.height());
    let mut buf = vec![0u8; (h * w * 3) as usize];
    for (i, p) in map.img.pixels().enumerate() {
        buf[i * 3..][..3].fill(p.0[0]);
    }
    for car in cars {
        for (x, y) in map.car_pixels(car) {
            set_px(&mut buf, w, h, x, y, [43, 127, 255]);
        }
        let (sa, ca) = car.theta.sin_cos();
        let (cx, cy) = map.position_to_pixels(car.x, car.y);
        for t in 0..8 {
            let ft = t as f64;
            set_px(
                &mut buf,
                w,
                h,
                cx + (ft * ca) as i32,
                cy - (ft * sa) as i32,
                [251, 44, 54],
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
