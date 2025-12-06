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

"""
 * Description:  Simplistic drone control:
 * - Stabilize the robot using the embedded sensors.
 * - Use PID technique to stabilize the drone roll/pitch/yaw.
 * - Use a cubic function applied on the vertical difference to stabilize the robot vertically.
 * - Stabilize the camera.
 * - Control the robot using the computer keyboard.
"""

import math
from controller import Robot, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor


def sign(x):
    """Return the sign of x: 1 if positive, -1 if negative, 0 if zero."""
    return 1 if x > 0 else (-1 if x < 0 else 0)


def clamp(value, low, high):
    """Clamp value between low and high."""
    return max(low, min(value, high))


# Initialize the robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get and enable devices
camera = robot.getDevice("camera")
camera.enable(timestep)

front_left_led = robot.getDevice("front left led")
front_right_led = robot.getDevice("front right led")

imu = robot.getDevice("inertial unit")
imu.enable(timestep)

gps = robot.getDevice("gps")
gps.enable(timestep)

compass = robot.getDevice("compass")
compass.enable(timestep)

gyro = robot.getDevice("gyro")
gyro.enable(timestep)

keyboard = Keyboard()
keyboard.enable(timestep)

camera_roll_motor = robot.getDevice("camera roll")
camera_pitch_motor = robot.getDevice("camera pitch")
# camera_yaw_motor = robot.getDevice("camera yaw")  # Not used in this example

# Get propeller motors and set them to velocity mode
front_left_motor = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor = robot.getDevice("rear left propeller")
rear_right_motor = robot.getDevice("rear right propeller")

motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]

for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(1.0)

# Display the welcome message
print("Start the drone...")

# Wait one second
while robot.step(timestep) != -1:
    if robot.getTime() > 1.0:
        break

# Display manual control message
print("You can control the drone with your computer keyboard:")
print("- 'up': move forward.")
print("- 'down': move backward.")
print("- 'right': turn right.")
print("- 'left': turn left.")
print("- 'shift + up': increase the target altitude.")
print("- 'shift + down': decrease the target altitude.")
print("- 'shift + right': strafe right.")
print("- 'shift + left': strafe left.")

# Constants, empirically found
K_VERTICAL_THRUST = 68.5  # with this thrust, the drone lifts
K_VERTICAL_OFFSET = 0.6   # Vertical offset where the robot actually targets to stabilize itself
K_VERTICAL_P = 3.0        # P constant of the vertical PID
K_ROLL_P = 50.0           # P constant of the roll PID
K_PITCH_P = 30.0          # P constant of the pitch PID

# Variables
target_altitude = 1.0  # The target altitude. Can be changed by the user.

# Main loop
while robot.step(timestep) != -1:
    time = robot.getTime()  # in seconds

    # Retrieve robot position using the sensors
    roll_pitch_yaw = imu.getRollPitchYaw()
    roll = roll_pitch_yaw[0]
    pitch = roll_pitch_yaw[1]
    
    gps_values = gps.getValues()
    altitude = gps_values[2]
    
    gyro_values = gyro.getValues()
    roll_velocity = gyro_values[0]
    pitch_velocity = gyro_values[1]

    # Blink the front LEDs alternatively with a 1 second rate
    led_state = int(time) % 2
    front_left_led.set(led_state)
    front_right_led.set(not led_state)

    # Stabilize the Camera by actuating the camera motors according to the gyro feedback
    camera_roll_motor.setPosition(-0.115 * roll_velocity)
    camera_pitch_motor.setPosition(-0.1 * pitch_velocity)

    # Transform the keyboard input to disturbances on the stabilization algorithm
    roll_disturbance = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance = 0.0
    
    key = keyboard.getKey()
    while key > 0:
        if key == Keyboard.UP:
            pitch_disturbance = -2.0
        elif key == Keyboard.DOWN:
            pitch_disturbance = 2.0
        elif key == Keyboard.RIGHT:
            yaw_disturbance = -1.3
        elif key == Keyboard.LEFT:
            yaw_disturbance = 1.3
        elif key == (Keyboard.SHIFT + Keyboard.RIGHT):
            roll_disturbance = -1.0
        elif key == (Keyboard.SHIFT + Keyboard.LEFT):
            roll_disturbance = 1.0
        elif key == (Keyboard.SHIFT + Keyboard.UP):
            target_altitude += 0.05
            print(f"target altitude: {target_altitude} [m]")
        elif key == (Keyboard.SHIFT + Keyboard.DOWN):
            target_altitude -= 0.05
            print(f"target altitude: {target_altitude} [m]")
        
        key = keyboard.getKey()

    # Compute the roll, pitch, yaw and vertical inputs
    roll_input = K_ROLL_P * clamp(roll, -1.0, 1.0) + roll_velocity + roll_disturbance
    pitch_input = K_PITCH_P * clamp(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance
    yaw_input = yaw_disturbance
    clamped_difference_altitude = clamp(target_altitude - altitude + K_VERTICAL_OFFSET, -1.0, 1.0)
    vertical_input = K_VERTICAL_P * math.pow(clamped_difference_altitude, 3.0)

    # Actuate the motors taking into consideration all the computed inputs
    front_left_motor_input = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input
    front_right_motor_input = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input
    rear_left_motor_input = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input
    rear_right_motor_input = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input
    
    front_left_motor.setVelocity(front_left_motor_input)
    front_right_motor.setVelocity(-front_right_motor_input)
    rear_left_motor.setVelocity(-rear_left_motor_input)
    rear_right_motor.setVelocity(rear_right_motor_input)
