use core_code::gnc::controls::pid::PidController;
use std::time::{SystemTime, Duration};
use std::thread;

#[test]
fn test_pid_initialization() {
    let pid = PidController::new(1.0, 0.1, 0.01, 10.0, -100.0, 100.0);
    assert_eq!(pid.kp, 1.0);
    assert_eq!(pid.ki, 0.1);
    assert_eq!(pid.kd, 0.01);
    assert_eq!(pid.setpoint, 10.0);
    assert_eq!(pid.output_min, -100.0);
    assert_eq!(pid.output_max, 100.0);
}

#[test]
fn test_pid_update() {
    let mut pid = PidController::new(1.0, 0.1, 0.01, 10.0, -100.0, 100.0);
    
    // Test with current value below setpoint
    let output1 = pid.update(5.0);
    assert!(output1 > 0.0); // Should give positive output to increase value
    
    // Test with current value above setpoint
    let output2 = pid.update(15.0);
    assert!(output2 < 0.0); // Should give negative output to decrease value
}

#[test]
fn test_pid_limits() {
    let mut pid = PidController::new(10.0, 0.0, 0.0, 10.0, -1.0, 1.0);
    
    // Test upper limit
    let output1 = pid.update(0.0); // Should try to output 100, but be limited to 1.0
    assert_eq!(output1, 1.0);
    
    // Test lower limit
    let output2 = pid.update(20.0); // Should try to output -100, but be limited to -1.0
    assert_eq!(output2, -1.0);
}

#[test]
fn test_pid_reset() {
    let mut pid = PidController::new(1.0, 0.1, 0.01, 10.0, -100.0, 100.0);
    
    // Create some history
    pid.update(5.0);
    pid.update(6.0);
    
    // Reset
    pid.reset();
    
    assert_eq!(pid.prev_error, 0.0);
    assert_eq!(pid.integral, 0.0);
}

#[test]
fn test_pid_timing() {
    let mut pid = PidController::new(1.0, 0.1, 0.01, 10.0, -100.0, 100.0);
    
    let output1 = pid.update(5.0);
    thread::sleep(Duration::from_millis(100));
    let output2 = pid.update(5.0);
    
    // The integral term should have accumulated over the time period
    assert!(output2 > output1);
}
