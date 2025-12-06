"""
 * Copyright 1996-2024 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

from controller import Robot, Camera, LED

from control import Control
from perception import Perception
from path_planner import PathPlanner


class DroneController():
    def __init__(self):
        print("Initialising Drone Controller...")

        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.front_left_led = self.robot.getDevice("front left led")
        self.front_right_led = self.robot.getDevice("front right led")
        
        self.control = Control(self.robot, self.timestep)
        self.perception = Perception(self.robot, self.timestep)
<<<<<<< Updated upstream
        # self.path_planner = PathPlanner(self.robot, self.timestep)


=======
        
        # World bounds for exploration (adjust based on your Webots world)
        world_bounds = {
            "x_min": -10,
            "x_max": 10,
            "y_min": -10,
            "y_max": 10,
            "z_min": 0.5,
            "z_max": 4.0
        }
        
        # Initialize simplified path planner (no transition model needed!)
        # print("[INIT] Initializing simple exploration path planner...")
        self.path_planner = SimplePathPlanner(
            robot=self.robot,
            timestep=self.timestep,
            perception=self.perception,
            world_bounds=world_bounds,
            exploration_resolution=1.0  # 1 meter grid cells
        )
        
>>>>>>> Stashed changes
        print("Drone Controller initialised successfully")


    def run(self):
<<<<<<< Updated upstream
        i = 0
        print("Running Drone Controller...")
        while self.robot.step(self.timestep) != -1:
            i += 1
            time = self.robot.getTime()  # in seconds
            
            if i == 20: print(self.perception.lidar.get_360_sectors(num_sectors=32)) 
            # controlling motors
            # start_state = self.perception.get_state_vector()
            # action = self.path_planner.get_best_action(start_state, num_steps=10)

            # velocities = self.control.stabilise(self.timestep, action)
            # self.control.process_signal(velocities)  # this should change propellers
            
            pass
=======
        step_counter = 0
        # print("Running Drone Controller with exploration...")
        
        # Wait for initial stabilization before starting exploration
        # print("[INIT] Waiting for sensors to stabilize...")
        for _ in range(10):
            self.robot.step(self.timestep)
        
        # print("[EXPLORATION] Starting exploration mode")
        
        while self.robot.step(self.timestep) != -1:
            self.perception.detect_items()
            step_counter += 1
            time = self.robot.getTime()  # in seconds
            
            try:
                # Get current state from perception
                state = self.perception.get_state_vector()
                position = state[:3]  # [x, y, z]
                yaw = state[5]  # yaw angle in radians
                
                # Update exploration map with current position and orientation
                self.path_planner.update_exploration(position, yaw)
                
                # Periodically print exploration stats
                if step_counter % 200 == 0:
                    stats = self.path_planner.get_exploration_stats()
                    # print(f"[EXPLORATION] Step {step_counter}: Explored {stats['exploration_percentage']:.1f}% "
                        #   f"({stats['explored_cells']}/{stats['total_cells']} cells)")
                    # print(f"[EXPLORATION] Current position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
                
                # Get desired position and orientation from path planner
                desired_pose = self.path_planner.get_desired_pose(state)
                desired_position = desired_pose['position']
                desired_orientation = desired_pose['orientation']
                
                # Compute control signals to reach desired position/orientation
                # Control system handles collision avoidance internally
                motor_velocities = self.control.compute_control(
                    current_state=state,
                    desired_position=desired_position,
                    desired_orientation=desired_orientation,
                    perception=self.perception,
                    dt=self.timestep/1000.0
                )
                
                # Apply motor velocities
                self.control.process_signal(motor_velocities)
                
                # Blink LEDs to show activity
                if step_counter % 100 == 0:
                    led_state = (step_counter // 100) % 2
                    self.front_left_led.set(led_state)
                    self.front_right_led.set(1 - led_state)
                    
            except Exception as e:
                # print(f"[ERROR] Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                # Continue with safe default action
                try:
                    default_action = [68.5, 68.5, 68.5, 68.5]  # Hover thrust
                    velocities = self.control.stabilise(self.timestep/1000, default_action)
                    self.control.process_signal(velocities)
                except:
                    pass
>>>>>>> Stashed changes

        print("Drone Controller stopped")


drone_controller = DroneController()
drone_controller.run()