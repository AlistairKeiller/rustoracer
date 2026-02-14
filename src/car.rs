#[derive(Clone)]
pub struct Car {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub velocity: f64,
    pub steering: f64,
}

pub struct Params {
    length_front: f64,
    length_rear: f64,
    steering_velocity_max: f64,
    a_max: f64,
    steering_max: f64,
    v_max: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            length_front: 0.15875,
            length_rear: 0.17145,
            steering_velocity_max: 3.2,
            a_max: 7.51,
            steering_max: 0.4189,
            v_max: 7.0,
        }
    }
}

impl Car {
    pub fn step(&mut self, steer_vel: f64, accel: f64, dt: f64, p: &Params) {
        let a = accel.clamp(-p.a_max, p.a_max);
        let sv = steer_vel.clamp(-p.steering_velocity_max, p.steering_velocity_max);
        let beta = (p.length_rear / (p.length_front + p.length_rear) * self.steering.tan()).atan();
        self.x += self.velocity * (self.theta + beta).cos() * dt;
        self.y += self.velocity * (self.theta + beta).sin() * dt;
        self.theta += self.velocity / p.length_rear * beta.sin() * dt;
        self.velocity = (self.velocity + a * dt).clamp(-p.v_max, p.v_max);
        self.steering = (self.steering + sv * dt).clamp(-p.steering_max, p.steering_max);
    }
}
