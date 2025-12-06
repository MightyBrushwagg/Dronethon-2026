"""
Model Predictive Control (MPC) for quadcopter position control.
Uses a simplified quadcopter model to predict future states and optimize control inputs.
"""

import numpy as np


class MPCController:
    """
    Model Predictive Controller for quadcopter position tracking.
    Uses a simplified linearized quadcopter model.
    """
    
    def __init__(self, horizon=10, dt=0.01, Q=None, R=None, Qf=None):
        """
        Initialize MPC controller.
        
        Args:
            horizon: Prediction horizon (number of steps)
            dt: Time step (seconds)
            Q: State cost matrix (12x12) - penalizes tracking error
            R: Control cost matrix (4x4) - penalizes control effort
            Qf: Final state cost matrix (12x12) - terminal cost
        """
        self.horizon = horizon
        self.dt = dt
        
        # Default cost matrices (tune these for performance)
        if Q is None:
            # Penalize position, orientation, velocity, and angular velocity errors
            # Balanced weights for better control considering all states
            Q = np.diag([10.0, 10.0, 50.0,  # position (x, y, z) - much higher z weight for altitude
                         0.5, 0.5, 0.5,    # orientation (roll, pitch, yaw) - moderate weights for stability
                         2.0, 2.0, 4.0,    # velocity (vx, vy, vz) - higher weights for velocity tracking
                         0.2, 0.2, 0.2])   # angular velocity (wx, wy, wz) - moderate weights for angular stability
        
        if R is None:
            # Penalize control effort (motor velocity changes)
            # Much higher penalty to strongly discourage large control signals
            R = np.diag([5.0, 5.0, 5.0, 5.0])  # Increased from 0.2 to strongly penalize large controls
        
        if Qf is None:
            # Terminal cost (same as Q for simplicity)
            Qf = Q.copy()
        
        self.Q = Q
        self.R = R
        self.Qf = Qf
        
        # Quadcopter physical parameters (simplified model)
        self.mass = 0.5  # kg (approximate)
        self.g = 9.81  # m/s^2
        self.hover_thrust = 69  # Base thrust for hover
        self.takeoff_thrust_boost = 5.0  # Extra thrust for takeoff (feedforward) - reduced for stability
        
        # Control input bounds
        self.u_min = 50.0  # Minimum motor velocity
        self.u_max = 85.0  # Maximum motor velocity
        
        # State: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        # Control: [motor_fl, motor_fr, motor_rl, motor_rr]
        
        # Warm start: store previous solution for faster convergence
        self.u_prev = None
        
    def predict_state(self, x, u, dt=None):
        """
        Predict next state using simplified quadcopter dynamics.
        
        Args:
            x: Current state [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
            u: Control input [motor_fl, motor_fr, motor_rl, motor_rr]
            dt: Time step (if None, uses self.dt)
        
        Returns:
            Next state prediction
        """
        if dt is None:
            dt = self.dt
        
        # Extract state components
        pos = x[:3]
        orient = x[3:6]
        vel = x[6:9]
        ang_vel = x[9:12]
        
        roll, pitch, yaw = orient
        
        # Simplified dynamics model
        # Motor mixing: convert motor velocities to forces and torques
        # Average thrust
        avg_thrust = np.mean(u)
        thrust_force = (avg_thrust - self.hover_thrust) * 0.15  # Increased scale factor from 0.1 for better altitude control
        
        # Roll/pitch/yaw torques from motor differences
        # Very reduced roll/pitch torque sensitivity to prevent aggressive control
        roll_torque = (u[1] + u[3] - u[0] - u[2]) * 0.002  # Further reduced from 0.005
        pitch_torque = (u[0] + u[1] - u[2] - u[3]) * 0.002  # Further reduced from 0.005
        yaw_torque = (u[0] + u[3] - u[1] - u[2]) * 0.01
        
        # Orientation dynamics (simplified)
        roll_dot = ang_vel[0]
        pitch_dot = ang_vel[1]
        yaw_dot = ang_vel[2]
        
        # Angular velocity dynamics (simplified)
        # Very reduced roll/pitch sensitivity to prevent aggressive control
        ang_vel_dot = np.array([
            roll_torque * 0.5,   # Roll angular acceleration - further reduced from 1.0
            pitch_torque * 0.5,  # Pitch angular acceleration - further reduced from 1.0
            yaw_torque * 1.0     # Yaw angular acceleration
        ])
        
        # Velocity dynamics in body frame
        # Rotate gravity to body frame
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        
        # Rotation matrix (simplified - using small angle approximation for roll/pitch)
        # For small angles: R ≈ [1, -yaw, pitch; yaw, 1, -roll; -pitch, roll, 1]
        # Gravity in world frame: [0, 0, -g]
        # Gravity in body frame (simplified)
        g_body = np.array([
            -self.g * sin_p,
            self.g * sin_r * cos_p,
            -self.g * cos_r * cos_p + thrust_force / self.mass
        ])
        
        # Rotate to world frame
        vel_dot_world = np.array([
            g_body[0] * cos_y - g_body[1] * sin_y,
            g_body[0] * sin_y + g_body[1] * cos_y,
            g_body[2]
        ])
        
        # Position dynamics
        pos_dot = vel
        
        # Integrate
        x_next = np.zeros(12)
        x_next[0:3] = pos + pos_dot * dt  # Position
        x_next[3:6] = orient + np.array([roll_dot, pitch_dot, yaw_dot]) * dt  # Orientation
        x_next[6:9] = vel + vel_dot_world * dt  # Velocity
        x_next[9:12] = ang_vel + ang_vel_dot * dt  # Angular velocity
        
        return x_next
    
    def compute_reference_trajectory(self, x0, x_target, horizon):
        """
        Compute reference trajectory from current state to target.
        Takes into account orientation and velocities for better control.
        
        Args:
            x0: Current state [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
            x_target: Target state (position [3] or full state [12])
            horizon: Number of steps
        
        Returns:
            Reference trajectory [horizon x 12]
        """
        ref_traj = np.zeros((horizon, 12))
        
        # Extract current state
        pos0 = x0[:3]
        orient0 = x0[3:6]  # [roll, pitch, yaw]
        vel0 = x0[6:9]     # [vx, vy, vz]
        ang_vel0 = x0[9:12]  # [wx, wy, wz]
        
        # Extract target state (handle both position-only and full state)
        if len(x_target) >= 12:
            # Full state target
            pos_target = x_target[:3]
            orient_target = x_target[3:6]
            vel_target = x_target[6:9]
            ang_vel_target = x_target[9:12]
        else:
            # Position-only target - set others to desired values
            pos_target = x_target[:3] if len(x_target) >= 3 else np.array([x_target[0], x_target[1], x_target[2]])
            orient_target = np.array([0.0, 0.0, orient0[2]])  # Level flight, keep current yaw
            vel_target = np.zeros(3)  # Stop at target
            ang_vel_target = np.zeros(3)  # No rotation
        
        # Desired velocity profile (smooth approach)
        total_distance = np.linalg.norm(pos_target - pos0)
        max_vel = 1.2  # m/s
        approach_time = min(horizon * self.dt, total_distance / max_vel if max_vel > 0 else horizon * self.dt)
        
        for k in range(horizon):
            t = k * self.dt
            alpha = min(1.0, t / approach_time) if approach_time > 0 else 1.0
            
            # Smooth interpolation (sigmoid-like)
            smooth_alpha = alpha * alpha * (3 - 2 * alpha)  # Smoothstep
            
            # Position reference - smooth trajectory
            ref_traj[k, :3] = pos0 + (pos_target - pos0) * smooth_alpha
            
            # Orientation reference - smooth transition to target orientation
            orient_error = orient_target - orient0
            # Wrap yaw error
            orient_error[2] = ((orient_error[2] + np.pi) % (2 * np.pi)) - np.pi
            ref_traj[k, 3:6] = orient0 + orient_error * smooth_alpha
            
            # Velocity reference - smooth velocity profile
            if k < horizon - 1:
                # Velocity from position trajectory
                pos_vel = (ref_traj[k+1, :3] - ref_traj[k, :3]) / self.dt
                # Blend with target velocity
                ref_traj[k, 6:9] = pos_vel * (1 - smooth_alpha) + vel_target * smooth_alpha
            else:
                ref_traj[k, 6:9] = vel_target  # Target velocity at end
            
            # Angular velocity reference - smooth transition to zero
            ref_traj[k, 9:12] = ang_vel0 * (1 - smooth_alpha) + ang_vel_target * smooth_alpha
        
        return ref_traj
    
    def cost_function(self, x_traj, u_traj, x_ref):
        """
        Compute cost for a trajectory.
        
        Args:
            x_traj: State trajectory [horizon x 4]
            u_traj: Control trajectory [horizon x 4]
            x_ref: Reference trajectory [horizon x 12]
        
        Returns:
            Total cost
        """
        cost = 0.0
        
        # Stage costs
        for k in range(self.horizon):
            state_error = x_traj[k] - x_ref[k]
            cost += state_error.T @ self.Q @ state_error
            
            # Control effort penalty (penalize large control signals)
            u_error = u_traj[k] - np.array([self.hover_thrust] * 4)  # Deviation from hover
            cost += u_traj[k].T @ self.R @ u_traj[k]  # Penalize absolute control
            cost += 0.5 * u_error.T @ u_error  # Additional penalty for deviation from hover
            
            # Penalize control changes (smoothness) - except for first step
            if k > 0:
                u_change = u_traj[k] - u_traj[k-1]
                cost += 0.3 * u_change.T @ u_change  # Penalize large control changes
        
        # Terminal cost
        state_error_final = x_traj[-1] - x_ref[-1]
        cost += state_error_final.T @ self.Qf @ state_error_final
        
        return cost
    
    def optimize_control(self, x0, x_target, u_init=None, max_iter=10):
        """
        Optimize control sequence using gradient descent.
        
        Args:
            x0: Current state [12]
            x_target: Target position [3] or state [12]
            u_init: Initial guess for control sequence [horizon x 4]
            max_iter: Maximum iterations for optimization (reduced for real-time)
        
        Returns:
            Optimal control sequence [horizon x 4]
        """
        # Initialize control sequence (warm start from previous solution)
        if u_init is None:
            if self.u_prev is not None:
                # Warm start: shift previous solution and repeat last control
                u_traj = np.zeros((self.horizon, 4))
                u_traj[:-1] = self.u_prev[1:]
                u_traj[-1] = self.u_prev[-1]
            else:
                u_traj = np.ones((self.horizon, 4)) * self.hover_thrust
        else:
            u_traj = u_init.copy()
        
        # Compute reference trajectory
        # Handle both position-only and full state targets
        if len(x_target) == 3:
            # Only position target, create full state target with desired orientation/velocity
            x_target_full = np.zeros(12)
            x_target_full[:3] = x_target  # Position
            x_target_full[3:6] = np.array([0.0, 0.0, x0[5]])  # Level flight, keep current yaw
            x_target_full[6:9] = np.zeros(3)  # Zero velocity at target
            x_target_full[9:12] = np.zeros(3)  # Zero angular velocity at target
        else:
            # Full state target provided
            if len(x_target) < 12:
                # Pad with zeros if needed
                x_target_full = np.zeros(12)
                x_target_full[:len(x_target)] = x_target
            else:
                x_target_full = x_target
        
        x_ref = self.compute_reference_trajectory(x0, x_target_full, self.horizon)
        
        # Gradient descent optimization
        learning_rate = 0.02  # Further reduced to prevent aggressive roll/pitch control
        best_cost = float('inf')
        best_u = u_traj.copy()
        
        for iteration in range(max_iter):
            # Simulate trajectory
            x_traj = np.zeros((self.horizon, 12))
            x_traj[0] = x0.copy()
            
            for k in range(self.horizon - 1):
                x_traj[k+1] = self.predict_state(x_traj[k], u_traj[k])
            
            # Compute cost
            cost = self.cost_function(x_traj, u_traj, x_ref)
            
            if cost < best_cost:
                best_cost = cost
                best_u = u_traj.copy()
            
            # Compute gradient (finite difference)
            grad = np.zeros_like(u_traj)
            eps = 0.05  # Reduced from 0.1 for smoother gradients (less aggressive)
            
            for k in range(self.horizon):
                for j in range(4):
                    u_perturbed = u_traj.copy()
                    u_perturbed[k, j] += eps
                    
                    # Simulate with perturbation
                    x_traj_pert = np.zeros((self.horizon, 12))
                    x_traj_pert[0] = x0.copy()
                    for kp in range(self.horizon - 1):
                        x_traj_pert[kp+1] = self.predict_state(x_traj_pert[kp], u_perturbed[kp])
                    
                    cost_pert = self.cost_function(x_traj_pert, u_perturbed, x_ref)
                    grad[k, j] = (cost_pert - cost) / eps
            
            # Update control sequence
            u_traj = u_traj - learning_rate * grad
            
            # Apply bounds
            u_traj = np.clip(u_traj, self.u_min, self.u_max)
            
            # Limit motor differences to prevent excessive roll/pitch
            # Keep motors more balanced to reduce roll/pitch control
            for k in range(self.horizon):
                avg_u = np.mean(u_traj[k])
                max_diff = 4.0  # Reduced from 6.0 - tighter constraint on motor differences
                for j in range(4):
                    if u_traj[k, j] > avg_u + max_diff:
                        u_traj[k, j] = avg_u + max_diff
                    elif u_traj[k, j] < avg_u - max_diff:
                        u_traj[k, j] = avg_u - max_diff
                
                # Also penalize large deviations from hover thrust
                for j in range(4):
                    hover_dev = abs(u_traj[k, j] - self.hover_thrust)
                    if hover_dev > 10.0:  # If too far from hover, pull back
                        if u_traj[k, j] > self.hover_thrust:
                            u_traj[k, j] = self.hover_thrust + 10.0
                        else:
                            u_traj[k, j] = self.hover_thrust - 10.0
            
            # Early stopping if cost is low enough
            if cost < 0.1:
                break
        
        # Store solution for warm start next time
        self.u_prev = best_u.copy()
        
        return best_u
    
    def compute_control(self, current_state, desired_position, desired_orientation=None, dt=None):
        """
        Compute MPC control action.
        Now takes into account orientation and velocities for better control.
        
        Args:
            current_state: Current state [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
            desired_position: Desired position [x, y, z]
            desired_orientation: Desired orientation [roll, pitch, yaw] (optional)
            dt: Time step (if None, uses self.dt)
        
        Returns:
            Control input [motor_fl, motor_fr, motor_rl, motor_rr]
        """
        if dt is not None:
            self.dt = dt
        
        # Build target state including orientation if provided
        if desired_orientation is not None and len(desired_orientation) >= 3:
            # Full state target with position and orientation
            target_state = np.zeros(12)
            target_state[:3] = desired_position[:3]
            target_state[3:6] = desired_orientation[:3]
            target_state[6:9] = np.zeros(3)  # Zero velocity at target
            target_state[9:12] = np.zeros(3)  # Zero angular velocity at target
        else:
            # Position-only target
            target_state = desired_position
        
        # Optimize control sequence (now considers full state)
        u_sequence = self.optimize_control(current_state, target_state)
        
        # Get first control action (receding horizon)
        u_control = u_sequence[0].copy()
        
        # FEEDFORWARD: Add takeoff boost when on ground
        current_altitude = current_state[2]
        desired_altitude = desired_position[2] if len(desired_position) >= 3 else current_altitude
        altitude_error = desired_altitude - current_altitude
        
        is_on_ground = current_altitude < 0.5  # Consider on ground if below 0.5m
        if is_on_ground and altitude_error > 0.05:  # Want to go up and on ground
            # Ramp down feedforward as altitude increases (prevents instability)
            altitude_factor = 1.0 - (current_altitude / 0.5)  # 1.0 at ground, 0.0 at 0.5m
            altitude_factor = max(0.2, altitude_factor)  # Keep at least 20% until higher
            
            # Add feedforward boost for takeoff - ramped
            takeoff_boost = self.takeoff_thrust_boost * min(1.5, altitude_error / 0.25) * altitude_factor
            if current_altitude < 0.2:  # Very close to ground
                takeoff_boost += 12.0 * altitude_factor
            if current_altitude < 0.15:  # Extremely close to ground
                takeoff_boost += 8.0 * altitude_factor
            u_control += takeoff_boost  # Add to all motors
        
        # Clamp to bounds
        u_control = np.clip(u_control, self.u_min, self.u_max)
        
        return u_control

