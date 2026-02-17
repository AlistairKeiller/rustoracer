// const G: f64 = 9.81;
// const V_KINEMATIC_THRESHOLD: f64 = 0.1; // m/s
// const MU: f64 = 1.0489;
// const C_SF: f64 = 4.718;
// const C_SR: f64 = 5.4562;
const LENGTH_FRONT: f64 = 0.15875;
const LENGTH_REAR: f64 = 0.17145;
const LENGTH_WHEELBASE: f64 = LENGTH_FRONT + LENGTH_REAR;
// const H: f64 = 0.074;
// const MASS: f64 = 3.74;
// const I_Z: f64 = 0.04712;
const STEERING_MIN: f64 = -0.4189;
const STEERING_MAX: f64 = 0.4189;
const STEERING_VELOCITY_MIN: f64 = -3.2;
const STEERING_VELOCITY_MAX: f64 = 3.2;
// const V_SWITCH: f64 = 7.319;
const A_MIN: f64 = -9.51;
const A_MAX: f64 = 9.51;
const V_MIN: f64 = -5.0;
const V_MAX: f64 = 20.0;
// const WIDTH: f64 = 0.31;
// const LENGTH: f64 = 0.58;

#[derive(Clone)]
pub struct Car {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub velocity: f64,
    pub steering: f64,
}

impl Car {
    pub fn step(&mut self, steer_vel: f64, accel: f64, dt: f64) {
        let a = accel.clamp(A_MIN, A_MAX);
        let sv = steer_vel.clamp(STEERING_VELOCITY_MIN, STEERING_VELOCITY_MAX);
        let beta = (LENGTH_REAR / LENGTH_WHEELBASE * self.steering.tan()).atan();
        let heading = self.theta + beta;
        self.x += self.velocity * heading.cos() * dt;
        self.y += self.velocity * heading.sin() * dt;
        self.theta += self.velocity / LENGTH_REAR * beta.sin() * dt;
        self.velocity = (self.velocity + a * dt).clamp(V_MIN, V_MAX);
        self.steering = (self.steering + sv * dt).clamp(STEERING_MIN, STEERING_MAX);
    }
}
