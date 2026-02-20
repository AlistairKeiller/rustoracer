mod car;
mod map;
#[cfg(feature = "ros")]
mod ros;
mod sim;
mod skeleton;

#[cfg(not(feature = "ros"))]
use crate::sim::Sim;

#[cfg(not(feature = "ros"))]
#[cfg_attr(feature = "show_images", show_image::main)]
fn main() {
    let mut sim = Sim::new("maps/berlin.yaml", 1);
    sim.reset(&[[0.0, 0.0, 0.0]]);
    for _ in 0..1000000 {
        let obs = sim.step(&[[0.0, 1.0]]);
        for (crashed, i) in obs.cols.iter().zip(0..obs.cols.len()) {
            if *crashed {
                sim.reset_single(&[0.0, 0.0, 0.0], i);
            }
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
