import math
import numpy as np
from controller import Robot, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor
from mpc_controller import MPCController

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
            
        # PID gains - increased for better stabilization during takeoff
        self.k_roll = {'p': 4.0, 'i': 0.0, 'd': 0.2}  # Increased for takeoff stability
        self.k_pitch = {'p': 6.0, 'i': 2.0, 'd': 0.6}  # Increased for takeoff stability
        self.k_yaw = {'p': 0.1, 'i': 0.0, 'd': 0.05}  # Reduced yaw gain to prevent spinning
        
        # integral accumulation
        self.integral = {'roll': 0, 'pitch': 0, 'yaw': 0, 'vertical': 0}
        self.prev_error = {'roll': 0, 'pitch': 0, 'yaw': 0, 'vertical': 0}
    
        self.k_vertical_thrust = 68.5;  # with this thrust, the drone lifts.
        self.k_vertical_offset = 0.6
        
        # Cascaded Control Structure (following MATLAB Parrot drone model):
        # Position Controller (Outer Loop) -> Velocity Controller (Inner Loop) -> Attitude Controller
        
        # Position Controller PID gains - converts position error to desired velocity
        self.k_pos = {
            'x': {'p': 0.3, 'i': 0.0, 'd': 0.0},   # Position to velocity for x
            'y': {'p': 0.3, 'i': 0.0, 'd': 0.0},   # Position to velocity for y
            'z': {'p': 4.0, 'i': 0.0, 'd': 0.0}    # Position to velocity for z (altitude) - much higher for takeoff
        }
        
        # Velocity Controller PID gains - converts velocity error to desired roll/pitch
        self.k_vel = {
            'x': {'p': 0.8, 'i': 0.0, 'd': 0.15},   # Velocity to pitch for x - reduced for less aggressive
            'y': {'p': 0.8, 'i': 0.0, 'd': 0.15},   # Velocity to roll for y - reduced for less aggressive
            'z': {'p': 12.0, 'i': 1.5, 'd': 2.5}   # Velocity to thrust for z (altitude) - increased for better altitude
        }
        
        # Altitude Controller PID gains - separate controller for collective thrust
        # Increased for better altitude tracking
        self.k_altitude = {'p': 10.0, 'i': 1.5, 'd': 2.5}  # Altitude error to thrust adjustment (increased further)
        
        # Takeoff feedforward parameters - balanced for stability
        self.takeoff_altitude_threshold = 0.4  # Consider on ground if below this (meters)
        self.takeoff_boost = 12.0  # Extra thrust for takeoff (feedforward) - reduced slightly for stability
        self.ground_altitude = 0.15  # Altitude considered as ground level
        
        # Attitude Controller PID gains - converts attitude error to motor commands
        # Reduced roll/pitch gains to be less aggressive
        self.kp_orient = {'roll': 1.5, 'pitch': 1.5, 'yaw': 2.0}  # Reduced roll/pitch for smoother control
        self.kd_orient = {'roll': 0.2, 'pitch': 0.2, 'yaw': 0.1}  # Reduced derivative for less aggressive damping
        
        # Integral accumulation for cascaded controllers
        self.integral_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}      # Position controller integrals
        self.integral_vel = {'x': 0.0, 'y': 0.0, 'z': 0.0}      # Velocity controller integrals
        self.integral_altitude = 0.0                             # Altitude controller integral
        self.prev_pos_error = {'x': 0.0, 'y': 0.0, 'z': 0.0}   # Previous position errors for derivative
        self.prev_vel_error = {'x': 0.0, 'y': 0.0, 'z': 0.0}    # Previous velocity errors for derivative
        self.prev_altitude_error = 0.0                          # Previous altitude error for derivative
        
        # Note: Obstacle avoidance moved to path planner
        # World bounds for boundary enforcement (will be set by drone controller)
        self.world_bounds = None
        
        # MPC Controller (optional - can be enabled for better trajectory tracking)
        self.use_mpc = False  # Set to True to use MPC, False to use cascaded PID
        self.mpc = None
        if self.use_mpc:
            dt_sec = self.timestep / 1000.0
            self.mpc = MPCController(horizon=15, dt=dt_sec)
            print("[CONTROL] MPC controller initialized")
        
        
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
        
        # Check if taking off - need stronger stabilization
        is_taking_off = altitude < 0.6  # During takeoff phase
        
        # stabilise camera by actuating the camera motors according to the gyro feedback.
        # Clamp gyro velocities to prevent extreme camera positions
        roll_vel_clamped = max(-5.0, min(5.0, roll_vel))  # Clamp to ±5 rad/s
        pitch_vel_clamped = max(-5.0, min(5.0, pitch_vel))  # Clamp to ±5 rad/s
        
        # Camera limits: roll -0.5 to 0.5, pitch -0.5 to 1.7
        camera_roll_pos = max(-0.5, min(0.5, -0.115 * roll_vel_clamped))
        camera_pitch_pos = max(-0.5, min(1.7, -0.1 * pitch_vel_clamped))
        self.camera_roll_motor.setPosition(camera_roll_pos)
        self.camera_pitch_motor.setPosition(camera_pitch_pos)
        
        # pid for stabilisation - stronger during takeoff to prevent flipping
        # Clamp roll/pitch inputs
        roll_clamped = max(-0.5, min(0.5, roll))  # Clamp to ±0.5 rad
        pitch_clamped = max(-0.5, min(0.5, pitch))  # Clamp to ±0.5 rad
        
        # Stronger stabilization during takeoff
        if is_taking_off:
            # More aggressive stabilization during takeoff to prevent rolling/flipping
            roll_vel_mult = 0.5  # Increased from 0.3
            pitch_vel_mult = 0.5  # Increased from 0.3
            roll_corr_mult = 1.5  # Multiplier for roll correction
            pitch_corr_mult = 1.5  # Multiplier for pitch correction
        else:
            roll_vel_mult = 0.3
            pitch_vel_mult = 0.3
            roll_corr_mult = 1.0
            pitch_corr_mult = 1.0
        
        roll_corr_raw = (self._pid(roll_clamped, 'roll', self.k_roll, dt) + roll_vel_clamped * roll_vel_mult) * roll_corr_mult
        pitch_corr_raw = (self._pid(pitch_clamped, 'pitch', self.k_pitch, dt) + pitch_vel_clamped * pitch_vel_mult) * pitch_corr_mult
        
        # DISABLE yaw control in stabilization - yaw is handled by the cascaded control structure
        yaw_corr = 0.0  # No yaw correction in stabilization
        
        # Clamp corrections - allow more during takeoff for stability
        if is_taking_off:
            roll_corr = max(-5.0, min(5.0, roll_corr_raw))  # Increased limits during takeoff
            pitch_corr = max(-5.0, min(5.0, pitch_corr_raw))
        else:
            roll_corr = max(-3.0, min(3.0, roll_corr_raw))
            pitch_corr = max(-3.0, min(3.0, pitch_corr_raw))
        
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
        
        # Enforce boundaries - clamp desired position to world bounds if available
        if self.world_bounds is not None:
            desired_position = np.array([
                np.clip(desired_position[0], self.world_bounds["x_min"], self.world_bounds["x_max"]),
                np.clip(desired_position[1], self.world_bounds["y_min"], self.world_bounds["y_max"]),
                np.clip(desired_position[2], self.world_bounds["z_min"], self.world_bounds["z_max"])
            ])
        
        # Use MPC controller if enabled
        if self.use_mpc and self.mpc is not None:
            try:
                # Update MPC time step
                self.mpc.dt = dt
                
                # Compute MPC control - now includes orientation and velocities
                mpc_control = self.mpc.compute_control(current_state, desired_position, desired_orientation, dt)
                
                # Apply stabilization corrections (MPC provides base control, stabilization fine-tunes)
                return self.stabilise(dt, mpc_control)
            except Exception as e:
                print(f"[MPC] Error in MPC controller, falling back to PID: {e}")
                # Fall through to PID controller
        
        # Extract current state (for PID controller)
        current_pos = current_state[:3]
        current_orient = current_state[3:6]  # [roll, pitch, yaw]
        current_vel = current_state[6:9]      # [vx, vy, vz]
        current_ang_vel = current_state[9:12]  # [wx, wy, wz]
        
        # Read sensors for stabilization
        roll, pitch, yaw_imu = self.imu.getRollPitchYaw()
        roll_vel, pitch_vel, yaw_vel = self.gyro.getValues()
        
        # ====================================================================
        # CASCADED CONTROL STRUCTURE (Following MATLAB Parrot Drone Model)
        # ====================================================================
        
        # STEP 1: POSITION CONTROLLER (Outer Loop)
        # Converts position error to desired velocity using PID
        pos_error = desired_position - current_pos
        
        # Boundary-aware position error scaling
        if self.world_bounds is not None:
            boundary_margin = 1.5
            dist_to_x_min = current_pos[0] - self.world_bounds["x_min"]
            dist_to_x_max = self.world_bounds["x_max"] - current_pos[0]
            dist_to_y_min = current_pos[1] - self.world_bounds["y_min"]
            dist_to_y_max = self.world_bounds["y_max"] - current_pos[1]
            
            x_scale = 1.0
            y_scale = 1.0
            if dist_to_x_min < boundary_margin:
                x_scale = max(0.3, dist_to_x_min / boundary_margin)
            elif dist_to_x_max < boundary_margin:
                x_scale = max(0.3, dist_to_x_max / boundary_margin)
            if dist_to_y_min < boundary_margin:
                y_scale = max(0.2, dist_to_y_min / boundary_margin)
            elif dist_to_y_max < boundary_margin:
                y_scale = max(0.2, dist_to_y_max / boundary_margin)
            
            pos_error[0] *= x_scale
            pos_error[1] *= y_scale
            
            # Reset integrals near boundaries
            boundary_reset_margin = 0.5
            if (current_pos[0] < self.world_bounds["x_min"] + boundary_reset_margin or 
                current_pos[0] > self.world_bounds["x_max"] - boundary_reset_margin):
                self.integral_pos['x'] = 0.0
            if (current_pos[1] < self.world_bounds["y_min"] + boundary_reset_margin or 
                current_pos[1] > self.world_bounds["y_max"] - boundary_reset_margin):
                self.integral_pos['y'] = 0.0
        
        # Position Controller PID: Position error -> Desired velocity
        # Update integrals
        self.integral_pos['x'] += pos_error[0] * dt
        self.integral_pos['y'] += pos_error[1] * dt
        self.integral_pos['z'] += pos_error[2] * dt
        
        # Anti-windup
        max_integral_pos = 1.0
        self.integral_pos['x'] = np.clip(self.integral_pos['x'], -max_integral_pos, max_integral_pos)
        self.integral_pos['y'] = np.clip(self.integral_pos['y'], -max_integral_pos, max_integral_pos)
        self.integral_pos['z'] = np.clip(self.integral_pos['z'], -max_integral_pos, max_integral_pos)
        
        # Calculate derivatives
        pos_error_derivative = {
            'x': (pos_error[0] - self.prev_pos_error['x']) / dt if dt > 0 else 0.0,
            'y': (pos_error[1] - self.prev_pos_error['y']) / dt if dt > 0 else 0.0,
            'z': (pos_error[2] - self.prev_pos_error['z']) / dt if dt > 0 else 0.0
        }
        
        # Position Controller output: Desired velocity
        desired_vel = np.array([
            self.k_pos['x']['p'] * pos_error[0] + 
            self.k_pos['x']['i'] * self.integral_pos['x'] + 
            self.k_pos['x']['d'] * pos_error_derivative['x'],
            self.k_pos['y']['p'] * pos_error[1] + 
            self.k_pos['y']['i'] * self.integral_pos['y'] + 
            self.k_pos['y']['d'] * pos_error_derivative['y'],
            self.k_pos['z']['p'] * pos_error[2] + 
            self.k_pos['z']['i'] * self.integral_pos['z'] + 
            self.k_pos['z']['d'] * pos_error_derivative['z']
        ])
        
        # Store previous errors
        self.prev_pos_error = {'x': pos_error[0], 'y': pos_error[1], 'z': pos_error[2]}
        
        # Limit desired velocities
        max_vel_xy = 1.2  # m/s
        max_vel_z = 0.8   # m/s
        vel_xy_mag = np.linalg.norm(desired_vel[:2])
        if vel_xy_mag > max_vel_xy:
            desired_vel[:2] = desired_vel[:2] * (max_vel_xy / vel_xy_mag)
        desired_vel[2] = np.clip(desired_vel[2], -max_vel_z, max_vel_z)
        
        # ====================================================================
        # STEP 2: VELOCITY CONTROLLER (Inner Loop) - Horizontal (x, y)
        # Converts velocity error to desired roll/pitch angles
        # ====================================================================
        vel_error_xy = desired_vel[:2] - current_vel[:2]
        
        # Add velocity damping to stabilize movement and prevent drift
        # If moving in wrong direction, add damping to reduce unwanted velocity
        velocity_damping = 0.5  # Damping factor for unwanted velocities
        if np.linalg.norm(desired_vel[:2]) < 0.1:  # If desired velocity is small
            # Damp out any unwanted movement
            vel_error_xy[0] -= current_vel[0] * velocity_damping
            vel_error_xy[1] -= current_vel[1] * velocity_damping
        
        # Reset velocity integrals if desired velocity is zero (prevent windup)
        if np.linalg.norm(desired_vel[:2]) < 0.05:
            self.integral_vel['x'] = 0.0
            self.integral_vel['y'] = 0.0
        
        # Update velocity controller integrals
        self.integral_vel['x'] += vel_error_xy[0] * dt
        self.integral_vel['y'] += vel_error_xy[1] * dt
        
        # Anti-windup
        max_integral_vel = 0.3  # Reduced to prevent overshoot
        self.integral_vel['x'] = np.clip(self.integral_vel['x'], -max_integral_vel, max_integral_vel)
        self.integral_vel['y'] = np.clip(self.integral_vel['y'], -max_integral_vel, max_integral_vel)
        
        # Velocity Controller output: Desired roll/pitch (in body frame)
        # Rotate velocity error to body frame
        cos_yaw = np.cos(yaw_imu)
        sin_yaw = np.sin(yaw_imu)
        vel_error_x_body = vel_error_xy[0] * cos_yaw + vel_error_xy[1] * sin_yaw
        vel_error_y_body = -vel_error_xy[0] * sin_yaw + vel_error_xy[1] * cos_yaw
        
        # Calculate velocity error derivatives (use world frame, then rotate to body frame)
        vel_error_derivative_world = {
            'x': (vel_error_xy[0] - self.prev_vel_error['x']) / dt if dt > 0 else 0.0,
            'y': (vel_error_xy[1] - self.prev_vel_error['y']) / dt if dt > 0 else 0.0
        }
        vel_error_derivative_x_body = vel_error_derivative_world['x'] * cos_yaw + vel_error_derivative_world['y'] * sin_yaw
        vel_error_derivative_y_body = -vel_error_derivative_world['x'] * sin_yaw + vel_error_derivative_world['y'] * cos_yaw
        
        # Velocity controller for pitch (forward/backward)
        # Positive pitch (nose down) = forward movement
        # Positive vel_error_x_body (need to move forward) -> positive pitch
        # Add deadband to prevent constant corrections for small errors
        vel_error_deadband = 0.05  # m/s - ignore very small velocity errors
        if abs(vel_error_x_body) < vel_error_deadband:
            pitch_from_vel = 0.0
        else:
            pitch_from_vel = (self.k_vel['x']['p'] * vel_error_x_body + 
                             self.k_vel['x']['i'] * (self.integral_vel['x'] * cos_yaw + self.integral_vel['y'] * sin_yaw) +
                             self.k_vel['x']['d'] * vel_error_derivative_x_body)
        
        # Velocity controller for roll (left/right)
        # Positive roll (right side down) = rightward movement
        # Positive vel_error_y_body (need to move right) -> positive roll
        if abs(vel_error_y_body) < vel_error_deadband:
            roll_from_vel = 0.0
        else:
            roll_from_vel = (self.k_vel['y']['p'] * vel_error_y_body + 
                            self.k_vel['y']['i'] * (-self.integral_vel['x'] * sin_yaw + self.integral_vel['y'] * cos_yaw) +
                            self.k_vel['y']['d'] * vel_error_derivative_y_body)
        
        # Store previous velocity errors (in world frame for next iteration)
        self.prev_vel_error['x'] = vel_error_xy[0]
        self.prev_vel_error['y'] = vel_error_xy[1]
        
        # Limit roll/pitch from velocity controller - reduced for less aggressive control
        max_roll_pitch_from_vel = 0.18  # ~10 degrees (reduced from 0.25 for smoother control)
        roll_from_vel = np.clip(roll_from_vel, -max_roll_pitch_from_vel, max_roll_pitch_from_vel)
        pitch_from_vel = np.clip(pitch_from_vel, -max_roll_pitch_from_vel, max_roll_pitch_from_vel)
        
        # Combine with desired orientation from path planner
        final_desired_roll = desired_orientation[0] + roll_from_vel
        final_desired_pitch = desired_orientation[1] + pitch_from_vel
        final_desired_yaw = desired_orientation[2]
        
        # ====================================================================
        # STEP 3: ALTITUDE CONTROLLER (Separate PD Controller)
        # Controls collective thrust based on altitude error
        # ====================================================================
        altitude_error = desired_position[2] - current_pos[2]
        
        # FEEDFORWARD: Add extra thrust for takeoff when on ground
        # This helps overcome gravity and initial inertia
        # Ramp down feedforward as altitude increases to prevent instability
        is_on_ground = current_pos[2] < self.takeoff_altitude_threshold
        takeoff_feedforward = 0.0
        if is_on_ground and altitude_error > 0.05:  # Want to go up and on ground
            # Ramp down feedforward as we gain altitude (prevents flipping)
            altitude_factor = 1.0 - (current_pos[2] / self.takeoff_altitude_threshold)  # 1.0 at ground, 0.0 at threshold
            altitude_factor = max(0.3, altitude_factor)  # Keep at least 30% until threshold
            
            # Add feedforward boost proportional to altitude error
            takeoff_feedforward = self.takeoff_boost * min(1.2, altitude_error / 0.3) * altitude_factor
            # Also add extra boost if very close to ground (but ramped)
            if current_pos[2] < self.ground_altitude:
                takeoff_feedforward += 8.0 * altitude_factor  # Ramped additional boost
            # Add constant boost when very close to ground
            if current_pos[2] < 0.15:
                takeoff_feedforward += 6.0 * altitude_factor  # Ramped constant boost
        
        # Update altitude integral
        self.integral_altitude += altitude_error * dt
        
        # Anti-windup - allow more integral when on ground for takeoff
        if is_on_ground:
            max_integral_altitude = 2.5  # Increased for better altitude tracking
        else:
            max_integral_altitude = 1.5  # Increased for better altitude tracking
        self.integral_altitude = np.clip(self.integral_altitude, -max_integral_altitude, max_integral_altitude)
        
        # Calculate derivative
        altitude_error_derivative = (altitude_error - self.prev_altitude_error) / dt if dt > 0 else 0.0
        
        # Altitude Controller: Altitude error -> Thrust adjustment
        # Also include vertical velocity control
        vertical_vel_error = desired_vel[2] - current_vel[2]
        altitude_thrust_adjustment = (
            self.k_altitude['p'] * altitude_error +
            self.k_altitude['i'] * self.integral_altitude +
            self.k_altitude['d'] * altitude_error_derivative +
            self.k_vel['z']['p'] * vertical_vel_error +  # Add vertical velocity control
            takeoff_feedforward  # Add feedforward for takeoff
        )
        
        # Store previous altitude error
        self.prev_altitude_error = altitude_error
        
        # ====================================================================
        # STEP 4: ATTITUDE CONTROLLER
        # Converts attitude error to motor commands
        # ====================================================================
        # Check if we're taking off (for stronger attitude control)
        is_taking_off = current_pos[2] < 0.5  # During takeoff phase
        
        # Check if we're moving (not at target)
        pos_error_magnitude = np.linalg.norm(pos_error[:2])
        is_moving = pos_error_magnitude > 0.15  # More than 15cm from target
        
        # When moving: maintain current yaw to prevent spinning
        # When at target: allow yaw changes for scanning
        if is_moving:
            # During movement: maintain current yaw (prevent unwanted spinning)
            final_desired_yaw = current_orient[2]  # Keep current yaw
        
        # Calculate attitude error
        orient_error = np.array([final_desired_roll, final_desired_pitch, final_desired_yaw]) - current_orient
        # Wrap yaw error to [-pi, pi]
        orient_error[2] = ((orient_error[2] + np.pi) % (2 * np.pi)) - np.pi
        
        # Attitude Controller: Attitude error -> Desired angular velocity
        # Stronger during takeoff to prevent rolling/flipping
        if is_taking_off:
            # Strong attitude control during takeoff to prevent instability
            roll_gain_mult = 2.0  # Increased for stability during takeoff
            pitch_gain_mult = 2.0  # Increased for stability during takeoff
        else:
            roll_gain_mult = 1.0
            pitch_gain_mult = 1.0
        
        desired_ang_vel = np.array([
            self.kp_orient['roll'] * orient_error[0] * roll_gain_mult - self.kd_orient['roll'] * current_ang_vel[0] * roll_gain_mult,
            self.kp_orient['pitch'] * orient_error[1] * pitch_gain_mult - self.kd_orient['pitch'] * current_ang_vel[1] * pitch_gain_mult,
            self.kp_orient['yaw'] * orient_error[2] - self.kd_orient['yaw'] * current_ang_vel[2]  # Always control yaw to prevent spinning
        ])
        
        # Clamp desired angular velocities
        if is_moving:
            # During movement: limit yaw angular velocity to prevent spinning
            desired_ang_vel = np.clip(desired_ang_vel, [-1.5, -1.5, -0.5], [1.5, 1.5, 0.5])
        else:
            # At target scanning: allow yaw rotation for scanning
            desired_ang_vel = np.clip(desired_ang_vel, [-1.5, -1.5, -2.0], [1.5, 1.5, 2.0])
        
        # ====================================================================
        # STEP 5: MOTOR COMMAND CALCULATION
        # Combine all control outputs into motor commands
        # ====================================================================
        # Base thrust from altitude controller
        base_thrust = self.k_vertical_thrust + altitude_thrust_adjustment
        
        # Attitude corrections from attitude controller
        # Stronger during takeoff to prevent rolling/flipping
        if is_taking_off:
            # Much stronger corrections during takeoff for stability
            roll_corr = orient_error[0] * 4.0 + desired_ang_vel[0] * 0.6  # Increased for takeoff stability
            pitch_corr = orient_error[1] * 4.0 + desired_ang_vel[1] * 0.6
        else:
            roll_corr = orient_error[0] * 1.5 + desired_ang_vel[0] * 0.25
            pitch_corr = orient_error[1] * 1.5 + desired_ang_vel[1] * 0.25
        yaw_corr = desired_ang_vel[2] * 0.8  # Only active when scanning at target
        
        # Clamp corrections - allow more during takeoff for stability
        if is_taking_off:
            roll_corr = np.clip(roll_corr, -6.0, 6.0)  # Increased limits during takeoff
            pitch_corr = np.clip(pitch_corr, -6.0, 6.0)
        else:
            roll_corr = np.clip(roll_corr, -4.0, 4.0)
            pitch_corr = np.clip(pitch_corr, -4.0, 4.0)
        yaw_corr = np.clip(yaw_corr, -3.0, 3.0)
        
        # Calculate motor velocities (quadcopter mixing)
        # Ensure balanced commands to prevent rolling/flipping
        motor_velocities = [
            base_thrust - roll_corr + pitch_corr - yaw_corr,  # front left
            base_thrust + roll_corr + pitch_corr + yaw_corr,  # front right
            base_thrust - roll_corr - pitch_corr + yaw_corr,  # rear left
            base_thrust + roll_corr - pitch_corr - yaw_corr,  # rear right
        ]
        
        # During takeoff, ensure motors are balanced (prevent large differences)
        if is_taking_off:
            # Limit motor differences to prevent instability
            avg_motor = np.mean(motor_velocities)
            max_diff = 8.0  # Maximum difference from average during takeoff
            for i in range(4):
                if motor_velocities[i] > avg_motor + max_diff:
                    motor_velocities[i] = avg_motor + max_diff
                elif motor_velocities[i] < avg_motor - max_diff:
                    motor_velocities[i] = avg_motor - max_diff
        
        # Clamp to reasonable range
        motor_velocities = [max(50.0, min(85.0, v)) for v in motor_velocities]
        
        # Apply stabilization corrections
        return self.stabilise(dt, motor_velocities)
        
        
    def STOP(self):
        """emergency STOP ALL MOTORS"""
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1.0)
        