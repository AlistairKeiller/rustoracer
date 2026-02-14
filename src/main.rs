mod car;
mod map;
#[cfg(feature = "ros")]
mod ros;
mod sim;

#[cfg(not(feature = "ros"))]
use crate::sim::Sim;

#[cfg(not(feature = "ros"))]
fn main() {
    let mut sim = Sim::new("maps/berlin.yaml", 1);
    let _obs = sim.reset(&[[0.0, 0.0, 0.0]]);
    for _ in 0..1000 {
        let (obs, col) = sim.step(&[[0.0, 1.0]]);
        if col[0] {
            println!(
                "collision at ({:.2}, {:.2})",
                obs.poses[0][0], obs.poses[0][1]
            );
            break;
        }
    }
}

#[cfg(feature = "ros")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bridge = ros::RosBridge {
        sim: sim::Sim::new("maps/berlin.yaml", 1),
        hz: 100.0,
    };
    bridge.spin(vec![[0.0, 0.0, 0.0]]).await
}
