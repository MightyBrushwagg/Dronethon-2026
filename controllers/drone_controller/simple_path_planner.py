"""
Simplified path planner for grid-based exploration.
No transition model needed - uses direct PID control to move toward unexplored cells.
"""

import numpy as np
from collections import defaultdict


HOVER_THRUST = 68.5
TAKEOFF_THRUST = 75.0  # Slightly higher thrust to ensure takeoff


class ExplorationMap:
    """
    Grid-based exploration map to track which areas the drone has explored.
    """
    def __init__(self, world_bounds, resolution=1.0):
        """
        Initialize exploration map.
        
        Args:
            world_bounds: Dict with 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
            resolution: Grid cell size in meters
        """
        self.resolution = resolution
        self.world_bounds = world_bounds
        
        # Calculate grid dimensions
        self.x_min = world_bounds.get("x_min", -20)
        self.x_max = world_bounds.get("x_max", 20)
        self.y_min = world_bounds.get("y_min", -20)
        self.y_max = world_bounds.get("y_max", 20)
        self.z_min = world_bounds.get("z_min", 0.1)
        self.z_max = world_bounds.get("z_max", 10.0)
        
        self.x_size = int((self.x_max - self.x_min) / resolution) + 1
        self.y_size = int((self.y_max - self.y_min) / resolution) + 1
        self.z_size = int((self.z_max - self.z_min) / resolution) + 1
        
        # Exploration grid: 0 = unexplored, >0 = exploration count
        self.exploration_grid = np.zeros((self.x_size, self.y_size, self.z_size), dtype=np.float32)
        
        print(f"[EXPLORATION] Initialized exploration map: {self.x_size}x{self.y_size}x{self.z_size} cells")
        print(f"[EXPLORATION] Resolution: {resolution}m, World bounds: X[{self.x_min}, {self.x_max}], Y[{self.y_min}, {self.y_max}], Z[{self.z_min}, {self.z_max}]")
    
    def position_to_grid(self, position):
        """Convert world position to grid indices."""
        x, y, z = position[0], position[1], position[2]
        
        # Check for NaN or Inf values and return default (center of grid)
        if np.isnan(x) or np.isinf(x) or np.isnan(y) or np.isinf(y) or np.isnan(z) or np.isinf(z):
            # Return center indices as default
            return self.x_size // 2, self.y_size // 2, self.z_size // 2
        
        x_idx = int((x - self.x_min) / self.resolution)
        y_idx = int((y - self.y_min) / self.resolution)
        z_idx = int((z - self.z_min) / self.resolution)
        
        # Clamp to valid grid bounds
        x_idx = max(0, min(self.x_size - 1, x_idx))
        y_idx = max(0, min(self.y_size - 1, y_idx))
        z_idx = max(0, min(self.z_size - 1, z_idx))
        
        return x_idx, y_idx, z_idx
    
    def grid_to_position(self, x_idx, y_idx, z_idx):
        """Convert grid indices to world position (center of cell)."""
        x = self.x_min + x_idx * self.resolution
        y = self.y_min + y_idx * self.resolution
        z = self.z_min + z_idx * self.resolution
        return np.array([x, y, z])
    
    def mark_explored(self, position, yaw=None):
        """
        Mark a position as explored.
        
        Args:
            position: [x, y, z] world position
            yaw: Yaw angle in radians (optional, for direction tracking)
        """
        x_idx, y_idx, z_idx = self.position_to_grid(position)
        # Increment exploration count
        self.exploration_grid[x_idx, y_idx, z_idx] += 1.0
    
    def get_exploration_value(self, position):
        """Get exploration value at position (0 = unexplored, higher = more explored)."""
        x_idx, y_idx, z_idx = self.position_to_grid(position)
        return self.exploration_grid[x_idx, y_idx, z_idx]
    
    def get_unexplored_targets(self, current_position, num_targets=3, min_distance=2.0, perception=None, 
                                min_safe_distance=1.0, obstacle_penalty=1000.0):
        """
        Find unexplored areas to explore, avoiding obstacles.
        
        Args:
            current_position: Current drone position [x, y, z]
            num_targets: Number of exploration targets to return
            min_distance: Minimum distance from current position
            perception: Perception module for obstacle detection (optional)
            min_safe_distance: Minimum safe distance from obstacles
            obstacle_penalty: Penalty score for targets near obstacles
        
        Returns:
            List of target positions [[x, y, z], ...] that avoid obstacles
        """
        x_idx, y_idx, z_idx = self.position_to_grid(current_position)
        
        # Get obstacle information if available
        obstacle_distances = None
        if perception is not None:
            try:
                obstacle_distances = perception.get_obstacle_distances()
            except:
                pass
        
        # Find all unexplored or less-explored cells
        candidates = []
        
        # Search in expanding radius from current position
        max_radius = max(self.x_size, self.y_size, self.z_size)
        
        for radius in range(1, max_radius):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        if dx*dx + dy*dy + dz*dz > radius * radius:
                            continue
                        
                        new_x_idx = x_idx + dx
                        new_y_idx = y_idx + dy
                        new_z_idx = z_idx + dz
                        
                        # Check bounds
                        if (new_x_idx < 0 or new_x_idx >= self.x_size or
                            new_y_idx < 0 or new_y_idx >= self.y_size or
                            new_z_idx < 0 or new_z_idx >= self.z_size):
                            continue
                        
                        # Check if unexplored or less explored
                        exploration_value = self.exploration_grid[new_x_idx, new_y_idx, new_z_idx]
                        
                        # Prefer less explored areas
                        target_pos = self.grid_to_position(new_x_idx, new_y_idx, new_z_idx)
                        distance = np.linalg.norm(target_pos - current_position)
                        
                        if distance >= min_distance:
                            # Check if target is safe from obstacles
                            obstacle_penalty = 0.0
                            if obstacle_distances:
                                # Calculate direction to target
                                direction_to_target = target_pos - current_position
                                direction_to_target[2] = 0  # Only horizontal
                                if np.linalg.norm(direction_to_target) > 0.01:
                                    direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
                                    
                                    # Check if target is in direction of obstacles
                                    # Simple heuristic: if target is in direction where obstacle is close, penalize
                                    if obstacle_distances.get("front", float('inf')) < min_safe_distance * 2:
                                        # Check if target is in front direction
                                        if direction_to_target[0] > 0.5:  # Mostly forward
                                            obstacle_penalty = obstacle_penalty
                                    
                                    if obstacle_distances.get("left", float('inf')) < min_safe_distance * 2:
                                        if direction_to_target[1] > 0.5:  # Mostly left
                                            obstacle_penalty = obstacle_penalty
                                    
                                    if obstacle_distances.get("right", float('inf')) < min_safe_distance * 2:
                                        if direction_to_target[1] < -0.5:  # Mostly right
                                            obstacle_penalty = obstacle_penalty
                            
                            # Score: lower exploration value is better, but prefer closer, and avoid obstacles
                            score = -exploration_value / (distance + 1.0) - obstacle_penalty
                            candidates.append((score, target_pos, exploration_value, obstacle_penalty))
            
            if len(candidates) >= num_targets * 10:  # Enough candidates
                break
        
        # Sort by score (best first) - obstacles will have very negative scores
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Deduplicate nearby targets and filter out obstacle-blocked targets
        selected = []
        for score, pos, exp_val, obs_penalty in candidates:
            # Skip if heavily penalized by obstacles
            if obs_penalty > 0:
                continue
            
            # Check if too close to already selected targets
            too_close = False
            for selected_pos in selected:
                if np.linalg.norm(pos - selected_pos) < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                selected.append(pos)
                if len(selected) >= num_targets:
                    break
        
        return selected
    
    def get_exploration_stats(self):
        """Get exploration statistics."""
        total_cells = self.x_size * self.y_size * self.z_size
        explored_cells = np.sum(self.exploration_grid > 0)
        exploration_percentage = (explored_cells / total_cells) * 100.0
        
        return {
            "total_cells": total_cells,
            "explored_cells": explored_cells,
            "exploration_percentage": exploration_percentage,
            "max_exploration": float(np.max(self.exploration_grid))
        }


class SimplePathPlanner:
    """
    Simplified path planner for grid exploration.
    Uses direct PID control to move toward unexplored grid cells.
    No transition model needed - much more efficient!
    """
    def __init__(self, robot, timestep, perception=None, world_bounds=None, exploration_resolution=1.0):
        """
        Initialize simple path planner.
        
        Args:
            robot: Webots robot instance
            timestep: Simulation timestep
            perception: Perception module instance (optional, for obstacle avoidance)
            world_bounds: Dict with 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
            exploration_resolution: Grid resolution for exploration map (meters)
        """
        self.robot = robot
        self.timestep = timestep
        self.perception = perception
        
        # World boundaries
        self.world_bounds = world_bounds if world_bounds else {
            "x_min": -20, "x_max": 20,
            "y_min": -20, "y_max": 20,
            "z_min": 0.5, "z_max": 10.0
        }
        
        # Initialize exploration map
        self.exploration_map = ExplorationMap(self.world_bounds, resolution=exploration_resolution)
        
        # Current target
        self.current_target = None
        self.target_update_counter = 0
        self.target_update_frequency = 100  # Update targets every N steps
        
        # Obstacle avoidance parameters
        self.min_safe_distance = 1.0  # Minimum safe distance from obstacles (meters)
        self.obstacle_penalty = 1000.0  # Penalty for targets near obstacles
        self.avoidance_margin = 0.5  # Additional margin when avoiding obstacles
        
        # PID gains for position control (tuned for smooth movement)
        self.kp_xy = 0.8  # Position gain for x/y
        self.kp_z = 1.0   # Position gain for z (altitude)
        self.kd_xy = 0.3  # Velocity damping for x/y (increased to prevent overshoot)
        self.kd_z = 0.2   # Velocity damping for z
        self.ki_xy = 0.01  # Integral term for x/y (small to prevent drift)
        
        # Integral accumulation for horizontal control
        self.integral_xy = np.array([0.0, 0.0])
        
        # Target altitude for exploration
        self.target_altitude = 2.0  # 2 meters above ground
        
        # State tracking
        self.has_taken_off = False
        self.takeoff_altitude = 1.5  # Consider taken off when above this
        
        # Set initial target to ensure takeoff
        # This will be updated once we get the first position reading
        self.current_target = None
        self.initial_target_set = False
        
        print("[SIMPLE PLANNER] Path planner initialized (no transition model needed!)")
    
    def update_exploration(self, position, yaw=None):
        """
        Update exploration map with current drone position.
        
        Args:
            position: Current position [x, y, z]
            yaw: Current yaw angle in radians (optional)
        """
        self.exploration_map.mark_explored(position, yaw)
        self.target_update_counter += 1
        
        # Check if we've taken off
        if not self.has_taken_off and position[2] > self.takeoff_altitude:
            self.has_taken_off = True
            print(f"[TAKEOFF] Drone has taken off! Altitude: {position[2]:.2f}m")
        
        # Periodically update exploration targets
        if self.target_update_counter >= self.target_update_frequency:
            self._update_exploration_targets(position)
            self.target_update_counter = 0
    
    def _update_exploration_targets(self, current_position):
        """Update exploration target based on unexplored areas, avoiding obstacles."""
        # Get unexplored targets (with obstacle avoidance)
        targets = self.exploration_map.get_unexplored_targets(
            current_position,
            num_targets=1,  # Just need one target at a time
            min_distance=2.0,
            perception=self.perception,
            min_safe_distance=self.min_safe_distance,
            obstacle_penalty=self.obstacle_penalty
        )
        
        if len(targets) > 0:
            # Set target altitude and clamp to world bounds
            target = targets[0].copy()
            target[2] = self.target_altitude  # Set target altitude
            
            # Clamp target to world bounds
            target[0] = np.clip(target[0], self.world_bounds["x_min"], self.world_bounds["x_max"])
            target[1] = np.clip(target[1], self.world_bounds["y_min"], self.world_bounds["y_max"])
            target[2] = np.clip(target[2], self.world_bounds["z_min"], self.world_bounds["z_max"])
            
            self.current_target = target
            
            stats = self.exploration_map.get_exploration_stats()
            print(f"[EXPLORATION] Updated target. Explored: {stats['exploration_percentage']:.1f}% "
                  f"({stats['explored_cells']}/{stats['total_cells']} cells)")
            print(f"[EXPLORATION] New target: ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")
        else:
            # No more unexplored areas nearby, expand search
            if self.current_target is None:
                # Set a default target to ensure we take off (clamped to bounds)
                center_x = (self.world_bounds["x_min"] + self.world_bounds["x_max"]) / 2
                center_y = (self.world_bounds["y_min"] + self.world_bounds["y_max"]) / 2
                self.current_target = np.array([center_x, center_y, self.target_altitude])
                print("[EXPLORATION] No unexplored areas nearby, using center target for takeoff")
    
    def get_desired_pose(self, state):
        """
        Get desired position and orientation for exploration.
        Returns either next location to explore, or same location with different orientation
        until the drone has properly reached the position.
        
        Args:
            state: Current state vector [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        
        Returns:
            dict with keys:
                - 'position': [x, y, z] desired position
                - 'orientation': [roll, pitch, yaw] desired orientation (roll and pitch should be 0 for level flight)
                - 'reached': bool indicating if current position has been fully explored
        """
        position = state[:3]  # [x, y, z]
        yaw = state[5]  # yaw angle
        
        # If no target set yet, set one above current position to ensure takeoff
        if self.current_target is None or not self.initial_target_set:
            self.current_target = np.array([position[0], position[1], self.target_altitude])
            self.initial_target_set = True
            print(f"[TAKEOFF] Setting initial target at altitude {self.target_altitude}m to ensure takeoff")
        
        # Clamp target to world bounds
        if self.current_target is not None:
            self.current_target = np.array([
                np.clip(self.current_target[0], self.world_bounds["x_min"], self.world_bounds["x_max"]),
                np.clip(self.current_target[1], self.world_bounds["y_min"], self.world_bounds["y_max"]),
                np.clip(self.current_target[2], self.world_bounds["z_min"], self.world_bounds["z_max"])
            ])
        
        # Check if we're outside bounds and create a target to return
        margin = 0.5  # Safety margin
        outside_bounds = (position[0] < self.world_bounds["x_min"] - margin or position[0] > self.world_bounds["x_max"] + margin or
                          position[1] < self.world_bounds["y_min"] - margin or position[1] > self.world_bounds["y_max"] + margin)
        
        if outside_bounds:
            # Force target to center of arena
            center_x = (self.world_bounds["x_min"] + self.world_bounds["x_max"]) / 2
            center_y = (self.world_bounds["y_min"] + self.world_bounds["y_max"]) / 2
            self.current_target = np.array([center_x, center_y, self.target_altitude])
            print(f"[BOUNDARY] Drone outside bounds! Position: ({position[0]:.2f}, {position[1]:.2f}), Returning to center: ({center_x:.1f}, {center_y:.1f})")
        
        # Check for obstacles and adjust target if necessary
        adjusted_target = self._adjust_target_for_obstacles(position, self.current_target.copy())
        self.current_target = adjusted_target
        
        # Calculate position error
        error = self.current_target - position
        horizontal_distance = np.linalg.norm(error[:2])
        vertical_distance = abs(error[2])
        
        # Check if we've reached the target position
        position_reached = horizontal_distance < 0.5 and vertical_distance < 0.3
        
        # Check if current position has been fully explored (all orientations)
        # For now, we'll consider a position "fully explored" if we've been there
        # In the future, we could track which yaw angles have been explored
        exploration_value = self.exploration_map.get_exploration_value(position)
        fully_explored = exploration_value > 3  # Visited at least 3 times
        
        if position_reached:
            # Reached target position
            if fully_explored:
                # Position fully explored, get new target
                self._update_exploration_targets(position)
                if self.current_target is None:
                    self.current_target = np.array([position[0], position[1], self.target_altitude])
                error = self.current_target - position
                horizontal_distance = np.linalg.norm(error[:2])
                vertical_distance = abs(error[2])
                position_reached = False
            else:
                # Position reached but not fully explored - explore different orientations
                # For now, just keep the same position but we could add orientation exploration
                pass
        
        # Determine desired orientation
        # For level flight, roll and pitch should be 0
        # Yaw can be set to face the target direction or explore different orientations
        desired_roll = 0.0
        desired_pitch = 0.0
        
        # If at target position but not fully explored, explore different yaw angles
        if position_reached and not fully_explored:
            # Rotate yaw to explore different directions
            # Simple strategy: increment yaw by 45 degrees each time
            target_yaw = (yaw + np.pi / 4) % (2 * np.pi)
        else:
            # Point yaw toward target
            if horizontal_distance > 0.1:
                target_direction = np.arctan2(error[1], error[0])
                target_yaw = target_direction
            else:
                # Close to target, maintain current yaw
                target_yaw = yaw
        
        desired_yaw = target_yaw
        
        return {
            'position': self.current_target.copy(),
            'orientation': np.array([desired_roll, desired_pitch, desired_yaw]),
            'reached': position_reached and fully_explored
        }
    
    def _adjust_target_for_obstacles(self, current_position, target_position):
        """
        Adjust target position to avoid obstacles detected by perception.
        
        Args:
            current_position: Current drone position [x, y, z]
            target_position: Original target position [x, y, z]
        
        Returns:
            Adjusted target position [x, y, z]
        """
        if self.perception is None:
            return target_position
        
        try:
            obstacle_distances = self.perception.get_obstacle_distances()
            adjusted_target = target_position.copy()
            
            # Calculate direction to target
            direction_to_target = target_position - current_position
            direction_to_target[2] = 0  # Only horizontal
            distance_to_target = np.linalg.norm(direction_to_target)
            
            if distance_to_target < 0.1:
                return target_position  # Already at target
            
            direction_to_target = direction_to_target / distance_to_target
            
            # Check obstacles in the direction we want to go
            # If obstacle is too close in our path, adjust target
            
            # Check front obstacle
            front_dist = obstacle_distances.get("front", float('inf'))
            if front_dist < self.min_safe_distance and direction_to_target[0] > 0.3:
                # Obstacle in front, adjust target to the side
                if obstacle_distances.get("left", float('inf')) > obstacle_distances.get("right", float('inf')):
                    # More space on left, move target left
                    adjusted_target[1] += self.avoidance_margin
                else:
                    # More space on right, move target right
                    adjusted_target[1] -= self.avoidance_margin
                
                # Also reduce forward component
                adjusted_target[0] -= self.avoidance_margin * 0.5
            
            # Check left obstacle
            left_dist = obstacle_distances.get("left", float('inf'))
            if left_dist < self.min_safe_distance and direction_to_target[1] > 0.3:
                # Obstacle on left, move target right
                adjusted_target[1] -= self.avoidance_margin
            
            # Check right obstacle
            right_dist = obstacle_distances.get("right", float('inf'))
            if right_dist < self.min_safe_distance and direction_to_target[1] < -0.3:
                # Obstacle on right, move target left
                adjusted_target[1] += self.avoidance_margin
            
            # Check vertical obstacles
            up_dist = obstacle_distances.get("up", float('inf'))
            if up_dist < self.min_safe_distance:
                # Obstacle above, lower target
                adjusted_target[2] = max(self.world_bounds["z_min"], adjusted_target[2] - self.avoidance_margin)
            
            down_dist = obstacle_distances.get("down", float('inf'))
            if down_dist < self.min_safe_distance:
                # Too close to ground, raise target
                adjusted_target[2] = min(self.world_bounds["z_max"], adjusted_target[2] + self.avoidance_margin)
            
            # Clamp to world bounds
            adjusted_target[0] = np.clip(adjusted_target[0], self.world_bounds["x_min"], self.world_bounds["x_max"])
            adjusted_target[1] = np.clip(adjusted_target[1], self.world_bounds["y_min"], self.world_bounds["y_max"])
            adjusted_target[2] = np.clip(adjusted_target[2], self.world_bounds["z_min"], self.world_bounds["z_max"])
            
            return adjusted_target
        
        except Exception as e:
            # If obstacle detection fails, return original target
            return target_position
    
    def get_best_action(self, state, num_steps=10):
        """
        Compatibility method - calls get_desired_pose.
        Kept for backward compatibility but should use get_desired_pose instead.
        """
        desired_pose = self.get_desired_pose(state)
        # Return a dummy action (will be replaced by control system)
        return np.array([HOVER_THRUST] * 4, dtype=np.float32)
    
    def get_exploration_stats(self):
        """Get current exploration statistics."""
        return self.exploration_map.get_exploration_stats()
    
    def update_goals(self, new_goal_positions):
        """Update goal positions (for compatibility)."""
        if len(new_goal_positions) > 0:
            self.current_target = np.array(new_goal_positions[0])
            self.current_target[2] = self.target_altitude  # Ensure correct altitude

