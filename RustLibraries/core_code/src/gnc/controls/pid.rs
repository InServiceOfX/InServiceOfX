use std::time::SystemTime;

#[derive(Debug)]
pub struct PidController {
    kp: f32,
    ki: f32,
    kd: f32,
    setpoint: f32,
    prev_error: f32,
    integral: f32,
    output_min: f32,
    output_max: f32,
    last_time: SystemTime,
}

impl PidController {
    pub fn new(kp: f32, ki: f32, kd: f32, setpoint: f32, output_min: f32, output_max: f32) -> Self {
        Self {
            kp,
            ki,
            kd,
            setpoint,
            prev_error: 0.0,
            integral: 0.0,
            output_min,
            output_max,
            last_time: SystemTime::now(),
        }
    }

    pub fn update(&mut self, current_value: f32) -> f32 {
        let current_time = SystemTime::now();
        let dt = current_time
            .duration_since(self.last_time)
            .unwrap_or_default()
            .as_secs_f32();
        self.last_time = current_time;

        let error = self.setpoint - current_value;
        
        // P term
        let p_term = self.kp * error;
        
        // I term
        self.integral += error * dt;
        let i_term = self.ki * self.integral;
        
        // D term
        let derivative = (error - self.prev_error) / dt;
        let d_term = self.kd * derivative;
        
        let mut output = p_term + i_term + d_term;
        output = output.clamp(self.output_min, self.output_max);
        
        self.prev_error = error;
        output
    }

    pub fn reset(&mut self) {
        self.prev_error = 0.0;
        self.integral = 0.0;
        self.last_time = SystemTime::now();
    }
}
