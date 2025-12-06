import math
from controller import Robot, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor
target_altitude = 1.0  # The target altitude. Don't go lower!

def sign(x):
    """Return the sign of x: 1 if positive, -1 if negative, 0 if zero."""
    return 1 if x > 0 else (-1 if x < 0 else 0)

class Drone(Robot):
    def __init__(self):
        super().__init__()
        self.fl_motor = robot.getDevice("front left propeller")
        self.fr_motor = robot.getDevice("front right propeller")
        self.rl_motor = robot.getDevice("rear left propeller")
        self.rr_motor = robot.getDevice("rear right propeller")
        self.motors = [fl_motor, fr_motor, rl_motor, rr_motor]
    
    def process_signal(self, vel):
        """receive 4 velocities for each propeller"""
        for motor, v in zip(self.motor, vel):
            motor.setPosition(float('inf'))
            motor.setVelocity(v)
        
        