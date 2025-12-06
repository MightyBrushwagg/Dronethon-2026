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
        self.path_planner = PathPlanner(self.robot, self.timestep)

        print("Drone Controller initialised successfully")

    def run(self):
        print("Running Drone Controller...")
        while self.robot.step(self.timestep) != -1:
            time = self.robot.getTime()  # in seconds
            pass

        print("Drone Controller stopped")


drone_controller = DroneController()
drone_controller.run()
