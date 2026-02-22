use crate::sim::Sim;
use futures::StreamExt;
use r2r::QosProfile;
use std::sync::{Arc, Mutex};
use std::time::Duration;

type AckDrive = r2r::ackermann_msgs::msg::AckermannDriveStamped;
type LaserScan = r2r::sensor_msgs::msg::LaserScan;
type Odom = r2r::nav_msgs::msg::Odometry;
type TFMsg = r2r::tf2_msgs::msg::TFMessage;
type TFS = r2r::geometry_msgs::msg::TransformStamped;
type Header = r2r::std_msgs::msg::Header;
type Stamp = r2r::builtin_interfaces::msg::Time;

pub struct RosBridge {
    pub sim: Sim,
    pub hz: f64,
}

impl RosBridge {
    pub async fn spin(mut self, poses: Vec<[f64; 3]>) -> Result<(), Box<dyn std::error::Error>> {
        let ctx = r2r::Context::create()?;
        let mut node = r2r::Node::create(ctx, "f1tenth_sim", "")?;
        let n = poses.len();
        let qos = QosProfile::default();

        self.sim.reset(&poses);

        let mut cmds: Vec<Arc<Mutex<[f64; 2]>>> = Vec::new();
        let mut scan_pubs = Vec::new();
        let mut odom_pubs = Vec::new();

        for i in 0..n {
            let ns = if n == 1 {
                String::new()
            } else {
                format!("agent{i}")
            };
            let t = |s: &str| {
                if ns.is_empty() {
                    format!("/{s}")
                } else {
                    format!("/{ns}/{s}")
                }
            };

            let cmd: Arc<Mutex<[f64; 2]>> = Arc::new(Mutex::new([0.0; 2]));
            let cmd_c = cmd.clone();
            let mut sub = node.subscribe::<AckDrive>(&t("drive"), qos.clone())?;
            tokio::spawn(async move {
                while let Some(msg) = sub.next().await {
                    *cmd_c.lock().unwrap() =
                        [msg.drive.steering_angle as f64, msg.drive.speed as f64]
                }
            });
            cmds.push(cmd);

            scan_pubs.push(node.create_publisher::<LaserScan>(&t("scan"), qos.clone())?);
            odom_pubs.push(node.create_publisher::<Odom>(&t("odom"), qos.clone())?);
        }

        let tf_pub = node.create_publisher::<TFMsg>("/tf", qos)?;
        let mut interval = tokio::time::interval(Duration::from_secs_f64(1.0 / self.hz));

        loop {
            interval.tick().await;
            node.spin_once(Duration::ZERO);

            let actions: Vec<[f64; 2]> = (0..n).map(|i| *cmds[i].lock().unwrap()).collect();

            let obs = self.sim.step(&actions);
            let stamp = now();

            let mut tfs = Vec::new();
            for i in 0..n {
                let [x, y, th, _, _, _, _] = obs.state[i];
                let v = self.sim.cars[i].velocity;
                let frame = if n == 1 {
                    "base_footprint".into()
                } else {
                    format!("agent{i}")
                };

                scan_pubs[i].publish(&build_scan(&obs.scans[i], &stamp, &frame, &self.sim))?;
                odom_pubs[i].publish(&build_odom(x, y, th, v, &stamp, &frame))?;

                tfs.push(build_tf(x, y, th, &stamp, "odom", &frame));
                tfs.push(build_tf(
                    0.0,
                    0.0,
                    0.0,
                    &stamp,
                    &frame,
                    &format!("{frame}/laser"),
                ));

                if obs.cols[i] {
                    self.sim.reset_single(&[0.0, 0.0, 0.0], i);
                }
            }
            tf_pub.publish(&TFMsg { transforms: tfs })?;
        }
    }
}

fn now() -> Stamp {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    Stamp {
        sec: d.as_secs() as i32,
        nanosec: d.subsec_nanos(),
    }
}

fn hdr(stamp: &Stamp, frame: &str) -> Header {
    Header {
        stamp: stamp.clone(),
        frame_id: frame.into(),
    }
}

fn yaw_q(y: f64) -> r2r::geometry_msgs::msg::Quaternion {
    r2r::geometry_msgs::msg::Quaternion {
        x: 0.0,
        y: 0.0,
        z: (y / 2.0).sin(),
        w: (y / 2.0).cos(),
    }
}

fn build_scan(ranges: &[f64], stamp: &Stamp, frame: &str, sim: &Sim) -> LaserScan {
    LaserScan {
        header: hdr(stamp, &format!("{frame}/laser")),
        angle_min: -(sim.fov / 2.0) as f32,
        angle_max: (sim.fov / 2.0) as f32,
        angle_increment: (sim.fov / (sim.n_beams - 1) as f64) as f32,
        range_min: 0.0,
        range_max: sim.max_range as f32,
        ranges: ranges.iter().map(|&r| r as f32).collect(),
        ..Default::default()
    }
}

fn build_odom(x: f64, y: f64, th: f64, v: f64, stamp: &Stamp, child: &str) -> Odom {
    use r2r::geometry_msgs::msg::*;
    Odom {
        header: hdr(stamp, "odom"),
        child_frame_id: child.into(),
        pose: PoseWithCovariance {
            pose: Pose {
                position: Point { x, y, z: 0.0 },
                orientation: yaw_q(th),
            },
            ..Default::default()
        },
        twist: TwistWithCovariance {
            twist: Twist {
                linear: Vector3 {
                    x: v,
                    y: 0.0,
                    z: 0.0,
                },
                ..Default::default()
            },
            ..Default::default()
        },
    }
}

fn build_tf(x: f64, y: f64, th: f64, stamp: &Stamp, parent: &str, child: &str) -> TFS {
    use r2r::geometry_msgs::msg::*;
    TFS {
        header: hdr(stamp, parent),
        child_frame_id: child.into(),
        transform: Transform {
            translation: Vector3 { x, y, z: 0.0 },
            rotation: yaw_q(th),
        },
    }
}
