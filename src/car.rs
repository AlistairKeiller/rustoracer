// pub const G: f64 = 9.81;
// pub const V_KINEMATIC_THRESHOLD: f64 = 0.1; // m/s
// pub const MU: f64 = 1.0489;
// pub const C_SF: f64 = 4.718;
// pub const C_SR: f64 = 5.4562;
pub const LENGTH_FRONT: f64 = 0.15875;
pub const LENGTH_REAR: f64 = 0.17145;
pub const LENGTH_WHEELBASE: f64 = LENGTH_FRONT + LENGTH_REAR;
// pub const H: f64 = 0.074;
// pub const MASS: f64 = 3.74;
// pub const I_Z: f64 = 0.04712;
pub const STEERING_MIN: f64 = -0.4189;
pub const STEERING_MAX: f64 = 0.4189;
pub const STEERING_VELOCITY_MIN: f64 = -3.2;
pub const STEERING_VELOCITY_MAX: f64 = 3.2;
// pub const V_SWITCH: f64 = 7.319;
pub const A_MIN: f64 = -9.51;
pub const A_MAX: f64 = 9.51;
pub const V_MIN: f64 = -5.0;
pub const V_MAX: f64 = 20.0;
pub const WIDTH: f64 = 0.31;
pub const LENGTH: f64 = 0.58;

#[derive(Clone)]
pub struct Car {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub velocity: f64,
    pub steering: f64,
}

type State = [f64; 5]; // [x, y, θ, v, δ]

fn add_scaled(base: &State, delta: &State, h: f64) -> State {
    std::array::from_fn(|i| base[i] + delta[i] * h)
}

fn rk4(y: &State, dt: f64, f: impl Fn(&State) -> State) -> State {
    let k1 = f(y);
    let k2 = f(&add_scaled(y, &k1, dt / 2.0));
    let k3 = f(&add_scaled(y, &k2, dt / 2.0));
    let k4 = f(&add_scaled(y, &k3, dt));
    std::array::from_fn(|i| y[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
}

fn bicycle_dynamics(s: &State, accel: f64, steer_vel: f64) -> State {
    let [_, _, theta, v, steering] = *s;
    let beta = (LENGTH_REAR / LENGTH_WHEELBASE * steering.tan()).atan();
    let heading = theta + beta;
    [
        v * heading.cos(),            // ẋ
        v * heading.sin(),            // ẏ
        v / LENGTH_REAR * beta.sin(), // θ̇
        accel,                        // v̇
        steer_vel,                    // δ̇
    ]
}

fn pid(steer: f64, speed: f64, current_speed: f64, current_steer: f64) -> (f64, f64) {
    let steer_diff = steer - current_steer;
    let sv = if steer_diff.abs() > 1e-4 {
        steer_diff.signum() * STEERING_VELOCITY_MAX
    } else {
        0.0
    };

    let vel_diff = speed - current_speed;
    let kp = if current_speed > 0.0 {
        if vel_diff > 0.0 {
            10.0 * A_MAX / V_MAX
        } else {
            10.0 * A_MAX / -V_MIN
        }
    } else {
        if vel_diff > 0.0 {
            2.0 * A_MAX / V_MAX
        } else {
            2.0 * A_MAX / -V_MIN
        }
    };

    (kp * vel_diff, sv)
}

impl Car {
    pub fn step(&mut self, steer: f64, speed: f64, dt: f64) {
        let (accel, steer_vel) = pid(steer, speed, self.velocity, self.steering);
        let a = accel.clamp(A_MIN, A_MAX);
        let sv = steer_vel.clamp(STEERING_VELOCITY_MIN, STEERING_VELOCITY_MAX);

        let state = [self.x, self.y, self.theta, self.velocity, self.steering];
        let [x, y, theta, v, s] = rk4(&state, dt, |s| bicycle_dynamics(s, a, sv));

        self.x = x;
        self.y = y;
        self.theta = theta;
        self.velocity = v.clamp(V_MIN, V_MAX);
        self.steering = s.clamp(STEERING_MIN, STEERING_MAX);
    }
}
