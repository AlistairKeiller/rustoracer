pub const G: f64 = 9.81;
pub const MU: f64 = 1.0489;
pub const C_SF: f64 = 4.718;
pub const C_SR: f64 = 5.4562;
pub const LF: f64 = 0.15875;
pub const LR: f64 = 0.17145;
pub const LWB: f64 = LF + LR;
pub const H: f64 = 0.074;
pub const MASS: f64 = 3.74;
pub const I_Z: f64 = 0.04712;
pub const STEER_MIN: f64 = -0.4189;
pub const STEER_MAX: f64 = 0.4189;
pub const STEER_VEL_MIN: f64 = -3.2;
pub const STEER_VEL_MAX: f64 = 3.2;
pub const V_SWITCH: f64 = 7.319;
pub const A_MAX: f64 = 9.51;
pub const V_MIN: f64 = -5.0;
pub const V_MAX: f64 = 20.0;
pub const WIDTH: f64 = 0.31;
pub const LENGTH: f64 = 0.58;

const V_KIN_THRESHOLD: f64 = 0.5;

/// State: [x, y, δ, vx, ψ, ψ̇, β]
type State = [f64; 7];

#[derive(Clone)]
pub struct Car {
    pub x: f64,
    pub y: f64,
    pub theta: f64,      // yaw angle ψ
    pub velocity: f64,   // longitudinal velocity vx
    pub steering: f64,   // front wheel angle δ
    pub yaw_rate: f64,   // ψ̇
    pub slip_angle: f64, // β
}

fn steering_constraint(steer: f64, sv: f64) -> f64 {
    let sv = sv.clamp(STEER_VEL_MIN, STEER_VEL_MAX);
    match sv {
        sv if sv < 0.0 && steer <= STEER_MIN => 0.0,
        sv if sv > 0.0 && steer >= STEER_MAX => 0.0,
        sv => sv,
    }
}

fn accel_constraint(vel: f64, a: f64) -> f64 {
    let a_max = if vel > V_SWITCH {
        A_MAX * V_SWITCH / vel
    } else {
        A_MAX
    };
    let a = a.clamp(-A_MAX, a_max);
    match a {
        a if a < 0.0 && vel <= V_MIN => 0.0,
        a if a > 0.0 && vel >= V_MAX => 0.0,
        a => a,
    }
}

fn kinematic_dynamics(s: &State, a: f64, sv: f64) -> State {
    let [_, _, steer, vx, yaw, _, _] = *s;
    let beta = (LR / LWB * steer.tan()).atan();
    let (sin_b, cos_sq) = (beta.sin(), steer.cos().powi(2));
    [
        vx * (yaw + beta).cos(),                          // ẋ
        vx * (yaw + beta).sin(),                          // ẏ
        sv,                                               // δ̇
        a,                                                // v̇x
        vx / LR * sin_b,                                  // ψ̇
        a / LWB * steer.tan() + vx / (LWB * cos_sq) * sv, // ψ̈
        0.0,                                              // β̇
    ]
}

fn dynamic_dynamics(s: &State, a: f64, sv: f64) -> State {
    let [_, _, steer, vx, yaw, yr, slip] = *s;

    let rear_load = G * LF + a * H;
    let front_load = G * LR - a * H;

    let yaw_acc = -MU * MASS / (vx * I_Z * LWB)
        * (LF * LF * C_SF * front_load + LR * LR * C_SR * rear_load)
        * yr
        + MU * MASS / (I_Z * LWB) * (LR * C_SR * rear_load - LF * C_SF * front_load) * slip
        + MU * MASS / (I_Z * LWB) * LF * C_SF * front_load * steer;

    let slip_rate = (MU / (vx * vx * LWB) * (C_SR * rear_load * LR - C_SF * front_load * LF) - 1.0)
        * yr
        - MU / (vx * LWB) * (C_SR * rear_load + C_SF * front_load) * slip
        + MU / (vx * LWB) * C_SF * front_load * steer;

    [
        vx * (slip + yaw).cos(), // ẋ
        vx * (slip + yaw).sin(), // ẏ
        sv,                      // δ̇
        a,                       // v̇x
        yr,                      // ψ̇
        yaw_acc,                 // ψ̈
        slip_rate,               // β̇
    ]
}

fn dynamics(s: &State, a: f64, sv: f64) -> State {
    if s[3].abs() < V_KIN_THRESHOLD {
        kinematic_dynamics(s, a, sv)
    } else {
        dynamic_dynamics(s, a, sv)
    }
}

fn rk4(y: &State, dt: f64, f: impl Fn(&State) -> State) -> State {
    let k1 = f(y);
    let k2 = f(&std::array::from_fn(|i| y[i] + k1[i] * dt / 2.0));
    let k3 = f(&std::array::from_fn(|i| y[i] + k2[i] * dt / 2.0));
    let k4 = f(&std::array::from_fn(|i| y[i] + k3[i] * dt));
    std::array::from_fn(|i| y[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
}

fn pid(target_steer: f64, target_speed: f64, vx: f64, steer: f64) -> (f64, f64) {
    let sv = if (target_steer - steer).abs() > 1e-4 {
        (target_steer - steer).signum() * STEER_VEL_MAX
    } else {
        0.0
    };
    let kp =
        if vx > 0.0 { 10.0 } else { 2.0 } * A_MAX / if target_speed > vx { V_MAX } else { -V_MIN };
    (kp * (target_speed - vx), sv)
}

impl Car {
    pub fn step(&mut self, steer: f64, speed: f64, dt: f64) {
        let (raw_a, raw_sv) = pid(steer, speed, self.velocity, self.steering);
        let a = accel_constraint(self.velocity, raw_a);
        let sv = steering_constraint(self.steering, raw_sv);

        let state: State = [
            self.x,
            self.y,
            self.steering,
            self.velocity,
            self.theta,
            self.yaw_rate,
            self.slip_angle,
        ];
        let [x, y, s, vx, yaw, yr, slip] = rk4(&state, dt, |s| dynamics(s, a, sv));

        self.x = x;
        self.y = y;
        self.steering = s.clamp(STEER_MIN, STEER_MAX);
        self.velocity = vx.clamp(V_MIN, V_MAX);
        self.theta = yaw;
        self.yaw_rate = yr;
        self.slip_angle = slip;
    }
}
