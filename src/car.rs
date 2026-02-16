use std::f64;

const G: f64 = 9.81;
const V_KINEMATIC_THRESHOLD: f64 = 0.1; // m/s
const MU: f64 = 1.0489;
const C_SF: f64 = 4.718;
const C_SR: f64 = 5.4562;
const LENGTH_FRONT: f64 = 0.15875;
const LENGTH_REAR: f64 = 0.17145;
const H: f64 = 0.074;
const MASS: f64 = 3.74;
const I_Z: f64 = 0.04712;
const STEERING_MIN: f64 = -0.4189;
const STEERING_MAX: f64 = 0.4189;
const STEERING_VELOCITY_MIN: f64 = -3.2;
const STEERING_VELOCITY_MAX: f64 = 3.2;
const V_SWITCH: f64 = 7.319;
const A_MAX: f64 = 9.51;
const V_MIN: f64 = -5.0;
const V_MAX: f64 = 20.0;
const WIDTH: f64 = 0.31;
const LENGTH: f64 = 0.58;

struct StateDerivatives {
    dx: f64,
    dy: f64,
    d_steering: f64,
    d_velocity: f64,
    d_yaw: f64,
    d_yaw_rate: f64,
    d_slip_angle: f64,
}

pub struct Params {
    pub mu: f64,
    pub c_sf: f64,
    pub c_sr: f64,
    pub lf: f64,
    pub lr: f64,
    pub h: f64,
    pub m: f64,
    pub i_z: f64,
    pub s_min: f64,
    pub s_max: f64,
    pub sv_min: f64,
    pub sv_max: f64,
    pub v_switch: f64,
    pub a_max: f64,
    pub v_min: f64,
    pub v_max: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            mu: MU,
            c_sf: C_SF,
            c_sr: C_SR,
            lf: LENGTH_FRONT,
            lr: LENGTH_REAR,
            h: H,
            m: MASS,
            i_z: I_Z,
            s_min: STEERING_MIN,
            s_max: STEERING_MAX,
            sv_min: STEERING_VELOCITY_MIN,
            sv_max: STEERING_VELOCITY_MAX,
            v_switch: V_SWITCH,
            a_max: A_MAX,
            v_min: V_MIN,
            v_max: V_MAX,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Car {
    pub x: f64,
    pub y: f64,
    pub steering: f64,
    pub velocity: f64,
    pub yaw: f64,
    pub yaw_rate: f64,
    pub slip_angle: f64,
}

/// Clamp steering velocity so the steering angle remains within
/// `[s_min, s_max]` and the rate stays within `[sv_min, sv_max]`.
fn constrain_steering_velocity(steering: f64, sv: f64, p: &Params) -> f64 {
    let mut sv = sv.clamp(p.sv_min, p.sv_max);
    if steering <= p.s_min && sv < 0.0 {
        sv = 0.0;
    } else if steering >= p.s_max && sv > 0.0 {
        sv = 0.0;
    }
    sv
}

/// Clamp longitudinal acceleration. Above `v_switch` the positive limit
/// is reduced proportionally; motion past the velocity bounds is prevented.
fn constrain_acceleration(velocity: f64, accel: f64, p: &Params) -> f64 {
    let pos_limit = if velocity > p.v_switch {
        p.a_max * p.v_switch / velocity
    } else {
        p.a_max
    };
    let mut a = accel.clamp(-p.a_max, pos_limit);
    if velocity <= p.v_min && a < 0.0 {
        a = 0.0;
    } else if velocity >= p.v_max && a > 0.0 {
        a = 0.0;
    }
    a
}

impl Car {
    fn derivatives(&self, sv: f64, acc: f64, p: &Params) -> StateDerivatives {
        let lwb = p.lf + p.lr;

        if self.velocity.abs() < V_KINEMATIC_THRESHOLD {
            // ── Kinematic bicycle model (reference point: center of gravity) ─

            let beta = (p.lr / lwb * self.steering.tan()).atan();

            let dx = self.velocity * (self.yaw + beta).cos();
            let dy = self.velocity * (self.yaw + beta).sin();
            let d_steering = sv;
            let d_velocity = acc;
            let d_yaw = self.velocity / lwb * self.steering.tan() * beta.cos();

            // Derivative of slip angle w.r.t. time
            let tan_s = self.steering.tan();
            let cos_s = self.steering.cos();
            let d_beta =
                (p.lr * sv) / (lwb * cos_s.powi(2) * (1.0 + (tan_s.powi(2) * p.lr / lwb).powi(2)));

            // Yaw acceleration
            let d_yaw_rate = (1.0 / lwb)
                * (acc * self.slip_angle.cos() * tan_s
                    - self.velocity * self.slip_angle.sin() * d_beta * tan_s
                    + self.velocity * self.slip_angle.cos() * sv / cos_s.powi(2));

            StateDerivatives {
                dx,
                dy,
                d_steering,
                d_velocity,
                d_yaw,
                d_yaw_rate,
                d_slip_angle: d_beta,
            }
        } else {
            // ── Dynamic single-track model ──────────────────────────────────

            let v = self.velocity;
            let mu = p.mu;
            let m = p.m;
            let lf = p.lf;
            let lr = p.lr;
            let iz = p.i_z;
            let c_sf = p.c_sf;
            let c_sr = p.c_sr;

            // Normal-load factors (gravity ± load transfer from acceleration)
            let fzf = (m * G * lr - m * acc * p.h) / lwb; // front
            let fzr = (m * G * lf + m * acc * p.h) / lwb; // rear

            // Position
            let dx = v * (self.slip_angle + self.yaw).cos();
            let dy = v * (self.slip_angle + self.yaw).sin();

            // Steering & velocity directly from constrained inputs
            let d_steering = sv;
            let d_velocity = acc;

            // Yaw
            let d_yaw = self.yaw_rate;

            // Yaw rate (moment balance)
            let d_yaw_rate = -mu * m / (v * iz * lwb)
                * (lf.powi(2) * c_sf * fzf + lr.powi(2) * c_sr * fzr)
                * self.yaw_rate
                + mu * m / (iz * lwb) * (lr * c_sr * fzr - lf * c_sf * fzf) * self.slip_angle
                + mu * m / (iz * lwb) * lf * c_sf * fzf * self.steering;

            // Slip-angle rate (lateral force balance)
            let d_slip_angle = (mu / (v.powi(2) * lwb) * (c_sr * fzr * lr - c_sf * fzf * lf) - 1.0)
                * self.yaw_rate
                - mu / (v * lwb) * (c_sr * fzr + c_sf * fzf) * self.slip_angle
                + mu / (v * lwb) * c_sf * fzf * self.steering;

            StateDerivatives {
                dx,
                dy,
                d_steering,
                d_velocity,
                d_yaw,
                d_yaw_rate,
                d_slip_angle,
            }
        }
    }

    /// Apply `StateDerivatives` scaled by `dt` to produce a new `Car` state.
    fn advance(&self, d: &StateDerivatives, dt: f64) -> Car {
        Car {
            x: self.x + d.dx * dt,
            y: self.y + d.dy * dt,
            steering: self.steering + d.d_steering * dt,
            velocity: self.velocity + d.d_velocity * dt,
            yaw: self.yaw + d.d_yaw * dt,
            yaw_rate: self.yaw_rate + d.d_yaw_rate * dt,
            slip_angle: self.slip_angle + d.d_slip_angle * dt,
        }
    }

    pub fn step(&mut self, steer_vel: f64, accel: f64, dt: f64, p: &Params) {
        // Constrain inputs once — they are held constant across all RK4 stages.
        let sv = constrain_steering_velocity(self.steering, steer_vel, p);
        let acc = constrain_acceleration(self.velocity, accel, p);

        // k1 = f(state_n)
        let k1 = self.derivatives(sv, acc, p);

        // k2 = f(state_n + dt/2 · k1)
        let s2 = self.advance(&k1, dt / 2.0);
        let k2 = s2.derivatives(sv, acc, p);

        // k3 = f(state_n + dt/2 · k2)
        let s3 = self.advance(&k2, dt / 2.0);
        let k3 = s3.derivatives(sv, acc, p);

        // k4 = f(state_n + dt · k3)
        let s4 = self.advance(&k3, dt);
        let k4 = s4.derivatives(sv, acc, p);

        // state_{n+1} = state_n + (dt / 6) · (k1 + 2·k2 + 2·k3 + k4)
        let dt6 = dt / 6.0;

        self.x += dt6 * (k1.dx + 2.0 * k2.dx + 2.0 * k3.dx + k4.dx);
        self.y += dt6 * (k1.dy + 2.0 * k2.dy + 2.0 * k3.dy + k4.dy);
        self.steering +=
            dt6 * (k1.d_steering + 2.0 * k2.d_steering + 2.0 * k3.d_steering + k4.d_steering);
        self.velocity +=
            dt6 * (k1.d_velocity + 2.0 * k2.d_velocity + 2.0 * k3.d_velocity + k4.d_velocity);
        self.yaw += dt6 * (k1.d_yaw + 2.0 * k2.d_yaw + 2.0 * k3.d_yaw + k4.d_yaw);
        self.yaw_rate +=
            dt6 * (k1.d_yaw_rate + 2.0 * k2.d_yaw_rate + 2.0 * k3.d_yaw_rate + k4.d_yaw_rate);
        self.slip_angle += dt6
            * (k1.d_slip_angle + 2.0 * k2.d_slip_angle + 2.0 * k3.d_slip_angle + k4.d_slip_angle);
    }
}
