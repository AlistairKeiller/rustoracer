mod car;
mod map;
mod sim;

use crate::sim::Sim;

fn main() {
    let mut sim = Sim::new("maps/berlin.yaml", 1);
    let _obs = sim.reset(&[[0.0, 0.0, 0.0]]);
    for _ in 0..1000 {
        let (obs, col) = sim.step(&[[0.0, 1.0]]); // [steer_vel, accel]
        if col[0] {
            println!(
                "collision at ({:.2}, {:.2})",
                obs.poses[0][0], obs.poses[0][1]
            );
            break;
        }
    }
}
