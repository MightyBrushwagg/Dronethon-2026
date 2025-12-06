import math
import numpy as np
from controller import Robot, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor

def sign(x):
    """Return the sign of x: 1 if positive, -1 if negative, 0 if zero."""
    return 1 if x > 0 else (-1 if x < 0 else 0)

class Control():
    def __init__(self, robot, timestep):
        self.robot = robot
        
        # motors
        self.fl_motor = self.robot.getDevice("front left propeller")
        self.fr_motor = self.robot.getDevice("front right propeller")
        self.rl_motor = self.robot.getDevice("rear left propeller")
        self.rr_motor = self.robot.getDevice("rear right propeller")
        self.motors = [self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor]
        
        self.timestep = timestep
        
        # camera control
        self.camera_roll_motor = robot.getDevice("camera roll")
        self.camera_pitch_motor = robot.getDevice("camera pitch")
        self.camera_roll_motor.setPosition(0)
        self.camera_pitch_motor.setPosition(0)
        self.camera_roll_motor.enableForceFeedback(self.timestep)
        self.camera_pitch_motor.enableForceFeedback(self.timestep)
        
        # sensors
        self.imu = self.robot.getDevice("inertial unit")
        self.imu.enable(timestep)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(timestep)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(timestep)
            
        # initalising motors
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1.0)
            
        # PID gains - reduced for smoother control
        self.k_roll = {'p': 3.0, 'i': 0.0, 'd': 0.1}
        self.k_pitch = {'p': 5.0, 'i': 2.0, 'd': 0.5}
        self.k_yaw = {'p': 0.1, 'i': 0.0, 'd': 0.05}  # Reduced yaw gain to prevent spinning
        
        # integral accumulation
        self.integral = {'roll': 0, 'pitch': 0, 'yaw': 0, 'vertical': 0}
        self.prev_error = {'roll': 0, 'pitch': 0, 'yaw': 0, 'vertical': 0}
    
        self.k_vertical_thrust = 68.5;  # with this thrust, the drone lifts.
        self.k_vertical_offset = 0.6
        
        # Position control PID gains - adjusted to reduce overshoot
        self.kp_pos_x = 0.2   # Position gain for x
        self.kp_pos_y = 0.2   # Position gain for y
        self.kp_pos_z = 0.2   # Position gain for z
        self.kd_pos_xy = 0.0  # Velocity damping for x/y
        self.kd_pos_z = 0.0   # Velocity damping for z
        self.ki_pos_xy = 0.0  # Integral term for x/y
        
        # Orientation control PID gains - faster yaw, less derivative
        self.kp_orient = {'roll': 1.2, 'pitch': 1.2, 'yaw': 2.0}  # Increased yaw for faster turning
        self.kd_orient = {'roll': 0.15, 'pitch': 0.15, 'yaw': 0.1}  # Reduced derivative to prevent overshoot
        
        # Integral accumulation for position control
        self.integral_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
        # Note: Obstacle avoidance moved to path planner
        # World bounds for boundary enforcement (will be set by drone controller)
        self.world_bounds = None
        
        
    def set_world_bounds(self, world_bounds):
        """Set world bounds for boundary enforcement."""
        self.world_bounds = world_bounds
    
    def process_signal(self, vel):
        """receive 4 velocities for each propeller"""
        # Motor signs from working C code: [1, -1, -1, 1]
        # front_left: positive, front_right: negative, rear_left: negative, rear_right: positive
        signs = [1, -1, -1, 1]
        for motor, v, sign in zip(self.motors, vel, signs):
            # motor.setPosition(float('inf'))
            signed_vel = sign * v
            clamped = max(-576, min(576, signed_vel))  # respect motor limits
            motor.setVelocity(clamped)
            
    def _pid(self, error, axis, gains, dt):
        self.integral[axis] += error * dt
        derivative = (error - self.prev_error[axis]) / dt if dt > 0 else 0
        self.prev_error[axis] = error
        return gains['p'] * error + gains['i'] * self.integral[axis] + gains['d'] * derivative
            
    def stabilise(self, dt, vel=None):
        if vel is None:
            vel = [self.k_vertical_thrust] * 4 # hover by default
            
        # read sensors
        roll, pitch, _ = self.imu.getRollPitchYaw()
        altitude = self.gps.getValues()[2]
        roll_vel, pitch_vel, yaw_vel = self.gyro.getValues()
        # Debug print removed for smoother operation
        
        # stabilise camera by actuating the camera motors according to the gyro feedback.
        # Clamp gyro velocities to prevent extreme camera positions
        roll_vel_clamped = max(-5.0, min(5.0, roll_vel))  # Clamp to ±5 rad/s
        pitch_vel_clamped = max(-5.0, min(5.0, pitch_vel))  # Clamp to ±5 rad/s
        
        # Camera limits: roll -0.5 to 0.5, pitch -0.5 to 1.7
        camera_roll_pos = max(-0.5, min(0.5, -0.115 * roll_vel_clamped))
        camera_pitch_pos = max(-0.5, min(1.7, -0.1 * pitch_vel_clamped))
        self.camera_roll_motor.setPosition(camera_roll_pos)
        self.camera_pitch_motor.setPosition(camera_pitch_pos)
        
        # pid for stabilisation - reduce gains to prevent overcorrection
        # Clamp roll/pitch inputs more aggressively
        roll_clamped = max(-0.5, min(0.5, roll))  # Clamp to ±0.5 rad
        pitch_clamped = max(-0.5, min(0.5, pitch))  # Clamp to ±0.5 rad
        
        # Use reduced PID corrections - scale down the final corrections
        # Reduce derivative influence to prevent overshoot
        roll_corr_raw = self._pid(roll_clamped, 'roll', self.k_roll, dt) + roll_vel_clamped * 0.3  # Reduced from 0.5
        pitch_corr_raw = self._pid(pitch_clamped, 'pitch', self.k_pitch, dt) + pitch_vel_clamped * 0.3  # Reduced from 0.5
        
        # Apply deadband to yaw control to prevent overcorrection
        # When called from compute_control(), yaw is already controlled, so reduce yaw correction here
        yaw_vel_deadband = 0.2  # Increased deadband - ignore small yaw velocities
        if abs(yaw_vel) > yaw_vel_deadband:
            # Only correct for large unwanted yaw velocities (spinning)
            # Use reduced gain to avoid fighting with compute_control's yaw control
            yaw_corr_raw = self._pid(yaw_vel, 'yaw', self.k_yaw, dt) * 0.3  # Reduce by 70%
        else:
            yaw_corr_raw = 0.0  # No correction for small yaw velocities
        
        # Clamp corrections to prevent extreme values (reduced for smoother control)
        roll_corr = max(-3.0, min(3.0, roll_corr_raw))  # Reduced from ±5.0
        pitch_corr = max(-3.0, min(3.0, pitch_corr_raw))  # Reduced from ±5.0
        yaw_corr = max(-1.0, min(1.0, yaw_corr_raw))  # Further reduced max yaw correction
        
        corrections = [
        -roll_corr + pitch_corr - yaw_corr,   # front left
        +roll_corr + pitch_corr + yaw_corr,   # front right
        -roll_corr - pitch_corr + yaw_corr,   # rear left
        +roll_corr - pitch_corr - yaw_corr,   # rear right
        ]
        
        # Clamp final corrections
        corrections = [max(-8.0, min(8.0, c)) for c in corrections]
        
        return [vel[i] + corrections[i] for i in range(4)]
    
    def compute_control(self, current_state, desired_position, desired_orientation, perception=None, dt=None):
        """
        Compute motor control signals to move drone to desired position and orientation.
        Includes collision avoidance.
        
        Args:
            current_state: Current state vector [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
            desired_position: [x, y, z] desired position
            desired_orientation: [roll, pitch, yaw] desired orientation
            perception: Perception module for obstacle detection (optional)
            dt: Time step in seconds (if None, uses self.timestep)
        
        Returns:
            List of 4 motor velocities [fl, fr, rl, rr]
        """
        if dt is None:
            dt = self.timestep / 1000.0
        
        # Extract current state
        current_pos = current_state[:3]
        current_orient = current_state[3:6]  # [roll, pitch, yaw]
        current_vel = current_state[6:9]      # [vx, vy, vz]
        current_ang_vel = current_state[9:12]  # [wx, wy, wz]
        
        # Read sensors for stabilization
        roll, pitch, yaw_imu = self.imu.getRollPitchYaw()
        roll_vel, pitch_vel, yaw_vel = self.gyro.getValues()
        
        # Enforce boundaries - clamp desired position to world bounds if available
        if self.world_bounds is not None:
            desired_position = np.array([
                np.clip(desired_position[0], self.world_bounds["x_min"], self.world_bounds["x_max"]),
                np.clip(desired_position[1], self.world_bounds["y_min"], self.world_bounds["y_max"]),
                np.clip(desired_position[2], self.world_bounds["z_min"], self.world_bounds["z_max"])
            ])
        
        # Position control (PID)
        pos_error = desired_position - current_pos
        
        # Boundary-aware velocity reduction - reduce velocity when approaching boundaries
        if self.world_bounds is not None:
            boundary_margin = 1.5  # Start reducing velocity when within 1.5m of boundary
            # Check distance to each boundary
            dist_to_x_min = current_pos[0] - self.world_bounds["x_min"]
            dist_to_x_max = self.world_bounds["x_max"] - current_pos[0]
            dist_to_y_min = current_pos[1] - self.world_bounds["y_min"]
            dist_to_y_max = self.world_bounds["y_max"] - current_pos[1]
            
            # Calculate velocity scaling factors for each axis
            x_scale = 1.0
            y_scale = 1.0
            
            # X axis boundary scaling
            if dist_to_x_min < boundary_margin:
                x_scale = max(0.3, dist_to_x_min / boundary_margin)  # Scale down as we approach
            elif dist_to_x_max < boundary_margin:
                x_scale = max(0.3, dist_to_x_max / boundary_margin)
            
            # Y axis boundary scaling (more aggressive for y since it's overshooting)
            if dist_to_y_min < boundary_margin:
                y_scale = max(0.2, dist_to_y_min / boundary_margin)  # More aggressive scaling
            elif dist_to_y_max < boundary_margin:
                y_scale = max(0.2, dist_to_y_max / boundary_margin)
            
            # Apply scaling to position error to reduce velocity toward boundaries
            pos_error[0] *= x_scale
            pos_error[1] *= y_scale
        
        # Reset integral if approaching boundaries to prevent windup
        if self.world_bounds is not None:
            boundary_reset_margin = 0.5  # Reset integral when very close to boundary
            if (current_pos[0] < self.world_bounds["x_min"] + boundary_reset_margin or 
                current_pos[0] > self.world_bounds["x_max"] - boundary_reset_margin):
                self.integral_pos['x'] = 0.0  # Reset x integral
            if (current_pos[1] < self.world_bounds["y_min"] + boundary_reset_margin or 
                current_pos[1] > self.world_bounds["y_max"] - boundary_reset_margin):
                self.integral_pos['y'] = 0.0  # Reset y integral (important for overshoot)
        
        # Update integral terms (with anti-windup)
        self.integral_pos['x'] += pos_error[0] * dt
        self.integral_pos['y'] += pos_error[1] * dt
        self.integral_pos['z'] += pos_error[2] * dt
        
        # Anti-windup
        max_integral = 0.8  # Reduced from 1.0 to prevent overshoot
        self.integral_pos['x'] = max(-max_integral, min(max_integral, self.integral_pos['x']))
        self.integral_pos['y'] = max(-max_integral, min(max_integral, self.integral_pos['y']))
        self.integral_pos['z'] = max(-max_integral, min(max_integral, self.integral_pos['z']))
        
        # Desired velocity from position control
        # Reduce derivative influence to prevent overshoot - scale down velocity damping
        # Use separate gains for y-axis to prevent overshoot
        desired_vel = np.array([
            self.kp_pos_x * pos_error[0] + self.ki_pos_xy * self.integral_pos['x'] - self.kd_pos_xy * current_vel[0] * 0.5,  # Reduce derivative
            self.kp_pos_y * pos_error[1] + self.ki_pos_xy * self.integral_pos['y'] - self.kd_pos_xy * current_vel[1] * 0.5,  # Reduced y gain
            self.kp_pos_z * pos_error[2] + self.ki_pos_xy * self.integral_pos['z'] - self.kd_pos_z * current_vel[2] * 0.5
        ])
        
        # Add distance-based velocity scaling to prevent overshoot near target
        distance_to_target = np.linalg.norm(pos_error[:2])
        if distance_to_target < 1.0:  # Within 1m of target
            # Scale down velocity as we approach target to prevent overshoot
            velocity_scale = max(0.3, distance_to_target / 1.0)  # Scale from 0.3 to 1.0
            desired_vel[:2] *= velocity_scale
        
        # Limit velocities - reduced for smoother, slower movement
        max_vel_xy = 0.8  # m/s (reduced from 1.5)
        max_vel_z = 0.6   # m/s (reduced from 1.0)
        vel_xy_mag = np.linalg.norm(desired_vel[:2])
        if vel_xy_mag > max_vel_xy:
            desired_vel[:2] = desired_vel[:2] * (max_vel_xy / vel_xy_mag)
        desired_vel[2] = np.clip(desired_vel[2], -max_vel_z, max_vel_z)
        
        # Note: Obstacle avoidance is now handled by the path planner
        # The path planner adjusts desired_position to avoid obstacles
        
        # Convert desired velocity to desired roll/pitch angles
        # Rotate to body frame
        cos_yaw = np.cos(yaw_imu)
        sin_yaw = np.sin(yaw_imu)
        desired_vx_body = desired_vel[0] * cos_yaw + desired_vel[1] * sin_yaw
        desired_vy_body = -desired_vel[0] * sin_yaw + desired_vel[1] * cos_yaw
        
        # Convert to desired roll/pitch angles from velocity (reduced for smoother control)
        desired_roll_from_vel = np.clip(desired_vy_body * 0.10, -0.15, 0.15)  # Max 0.15 rad (~8.6 deg)
        desired_pitch_from_vel = np.clip(desired_vx_body * 0.10, -0.15, 0.15)
        
        # Combine desired orientation from path planner with velocity-based corrections
        # Path planner provides desired orientation, but we add velocity-based adjustments
        final_desired_roll = desired_orientation[0] + desired_roll_from_vel
        final_desired_pitch = desired_orientation[1] + desired_pitch_from_vel
        final_desired_yaw = desired_orientation[2]
        
        # Orientation control (PID)
        orient_error = np.array([final_desired_roll, final_desired_pitch, final_desired_yaw]) - current_orient
        # Wrap yaw error to [-pi, pi]
        orient_error[2] = ((orient_error[2] + np.pi) % (2 * np.pi)) - np.pi
        
        # Check if we're moving (not at target) - if so, disable yaw control
        # Only allow yaw changes when scanning at target
        pos_error_magnitude = np.linalg.norm(pos_error[:2])
        is_moving = pos_error_magnitude > 0.15  # More than 15cm from target
        
        # Desired angular velocities from orientation control
        # For yaw, we want to correct the error, so positive error should create positive angular velocity
        # Reduced derivative influence to prevent overshoot
        desired_ang_vel = np.array([
            self.kp_orient['roll'] * orient_error[0] - self.kd_orient['roll'] * current_ang_vel[0] * 0.5,  # Reduce derivative influence
            self.kp_orient['pitch'] * orient_error[1] - self.kd_orient['pitch'] * current_ang_vel[1] * 0.5,
            # Only apply yaw control if not moving (at target scanning)
            (self.kp_orient['yaw'] * orient_error[2] - self.kd_orient['yaw'] * current_ang_vel[2] * 0.3) if not is_moving else 0.0
        ])
        
        # Clamp desired angular velocities - allow faster yaw only when scanning
        if is_moving:
            # During movement: no yaw rotation
            desired_ang_vel = np.clip(desired_ang_vel, [-1.0, -1.0, 0.0], [1.0, 1.0, 0.0])
        else:
            # At target scanning: allow yaw rotation
            desired_ang_vel = np.clip(desired_ang_vel, [-1.0, -1.0, -2.0], [1.0, 1.0, 2.0])
        
        # Base thrust calculation
        base_thrust = self.k_vertical_thrust
        
        # Vertical velocity correction - increased to ensure altitude control works
        # With low position gains, need higher multiplier to achieve altitude
        vertical_correction = desired_vel[2] * 5.0  # Increased to ensure altitude control
        vertical_correction = np.clip(vertical_correction, -8.0, 8.0)
        
        # Roll/pitch corrections from orientation control
        roll_corr = orient_error[0] * 2.0 + desired_ang_vel[0] * 0.3
        pitch_corr = orient_error[1] * 2.0 + desired_ang_vel[1] * 0.3
        # Yaw correction - only when scanning (not moving)
        # If moving, desired_ang_vel[2] will be 0, so yaw_corr will be 0
        yaw_corr = desired_ang_vel[2] * 0.8  # Only active when scanning at target
        
        # Clamp corrections
        roll_corr = np.clip(roll_corr, -5.0, 5.0)
        pitch_corr = np.clip(pitch_corr, -5.0, 5.0)
        yaw_corr = np.clip(yaw_corr, -3.0, 3.0)
        
        # Calculate motor velocities
        motor_velocities = [
            base_thrust - roll_corr + pitch_corr - yaw_corr + vertical_correction,  # front left
            base_thrust + roll_corr + pitch_corr + yaw_corr + vertical_correction,  # front right
            base_thrust - roll_corr - pitch_corr + yaw_corr + vertical_correction,  # rear left
            base_thrust + roll_corr - pitch_corr - yaw_corr + vertical_correction,  # rear right
        ]
        
        # Clamp to reasonable range
        motor_velocities = [max(50.0, min(85.0, v)) for v in motor_velocities]
        
        # Apply stabilization corrections
        return self.stabilise(dt, motor_velocities)
        
        
    def STOP(self):
        """emergency STOP ALL MOTORS"""
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1.0)
        