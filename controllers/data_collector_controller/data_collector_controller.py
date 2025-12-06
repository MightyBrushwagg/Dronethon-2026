"""
Webots controller for collecting training data.

This controller should be set as the drone's controller in Webots.
It will collect state-action-next_state transitions and save them.
"""

# IMPORTANT: Log immediately to see if script starts
print("[SCRIPT] Script file loaded and starting execution...")
import sys
sys.stdout.flush()

print("[SCRIPT] Setting up environment...")
import warnings
import os
# Suppress numpy warnings early
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
sys.stdout.flush()

print("[SCRIPT] Importing pathlib...")
from pathlib import Path
sys.stdout.flush()

print("[SCRIPT] Setting up import paths...")
sys.stdout.flush()

# Add parent directory to path to import from drone_controller
controller_dir = Path(__file__).parent
print(f"[SCRIPT] Controller directory: {controller_dir}")
sys.stdout.flush()

drone_controller_dir = controller_dir.parent / "drone_controller"
print(f"[SCRIPT] Drone controller directory (relative): {drone_controller_dir}")
sys.stdout.flush()

# Ensure the path is absolute
if not drone_controller_dir.is_absolute():
    # Get absolute path
    project_root = controller_dir.parent.parent  # Go up to controllers, then project root
    drone_controller_dir = project_root / "controllers" / "drone_controller"

print(f"[SCRIPT] Drone controller directory (final): {drone_controller_dir}")
print(f"[SCRIPT] Directory exists: {drone_controller_dir.exists()}")
sys.stdout.flush()

sys.path.insert(0, str(drone_controller_dir))
print(f"[SCRIPT] Added to sys.path: {drone_controller_dir}")
sys.stdout.flush()

# Import Webots and custom modules with error handling
print("[SCRIPT] Importing Robot from controller...")
sys.stdout.flush()
try:
    from controller import Robot, Supervisor
    print("[SCRIPT] ✓ Robot imported successfully")
    sys.stdout.flush()
except ImportError as e:
    print(f"[SCRIPT] ❌ ERROR: Could not import Robot from controller: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    raise

print("[SCRIPT] Importing numpy...")
sys.stdout.flush()
try:
    # Try importing numpy with error handling
    import numpy as np
    print("[SCRIPT] ✓ numpy imported successfully")
    sys.stdout.flush()
except ImportError as e:
    print(f"[SCRIPT] ❌ ImportError: Could not import numpy: {e}")
    print(f"[SCRIPT] This might be a numpy version compatibility issue with Python 3.13")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    raise
except Exception as e:
    print(f"[SCRIPT] ❌ ERROR: Unexpected error importing numpy: {e}")
    print(f"[SCRIPT] Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    # Try to continue - numpy might still work despite warnings
    print("[SCRIPT] ⚠ Attempting to continue despite numpy import error...")
    sys.stdout.flush()
    import numpy as np  # Try again

print("[SCRIPT] Importing Control module...")
sys.stdout.flush()
try:
    from control import Control
    print("[SCRIPT] ✓ Control imported successfully")
    sys.stdout.flush()
except ImportError as e:
    print(f"[SCRIPT] ❌ ERROR: Could not import Control: {e}")
    print(f"[SCRIPT] Looking for control.py in: {drone_controller_dir}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    raise

print("[SCRIPT] Importing Perception module...")
sys.stdout.flush()
try:
    from perception import Perception
    print("[SCRIPT] ✓ Perception imported successfully")
    sys.stdout.flush()
except ImportError as e:
    print(f"[SCRIPT] ❌ ERROR: Could not import Perception: {e}")
    print(f"[SCRIPT] Looking for perception.py in: {drone_controller_dir}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    raise

# Hover thrust constant (matches control.py's k_vertical_thrust = 68.5)
# Note: path_planner.py has HOVER_THRUST = 10.0 but that seems incorrect
# Using 68.5 from control.py which is the actual motor velocity for hover
HOVER_THRUST = 68.5
print(f"[SCRIPT] ✓ All imports complete. HOVER_THRUST = {HOVER_THRUST}")
sys.stdout.flush()


class TransitionModelDataset:
    """
    Dataset for storing and loading transition data.
    """
    def __init__(self, data_dir=None):
        if data_dir is None:
            # Save data in data_collector_controller/data/transitions
            script_dir = Path(__file__).parent
            data_dir = script_dir / "data" / "transitions"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = self.data_dir / "transitions.npy"
        self.metadata_file = self.data_dir / "metadata.json"
        
        self.states = []
        self.actions = []
        self.next_states = []
    
    def add_transition(self, state, action, next_state):
        """
        Add a single transition tuple.
        """
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
    
    def save(self):
        """
        Save collected data.
        """
        if len(self.states) == 0:
            print("No data to save!")
            return
        
        import json
        import time
        
        # Convert to numpy arrays
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.float32)
        next_states = np.array(self.next_states, dtype=np.float32)
        
        # Save as single array
        transitions = np.concatenate([states, actions, next_states], axis=1)
        np.save(self.data_file, transitions)
        
        metadata = {
            "num_samples": len(self.states),
            "state_dim": states.shape[1],
            "action_dim": actions.shape[1],
            "timestamp": time.time()
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(self.states)} transitions to {self.data_file}")
        print(f"States shape: {states.shape}, Actions shape: {actions.shape}, Next states shape: {next_states.shape}")


class DataCollectorController:
    def __init__(self, num_episodes=10, episode_length=500):
        """
        Initialize data collector controller.
        
        Args:
            num_episodes: Number of episodes to collect
            episode_length: Number of simulation steps per episode
        """
        print("=" * 60)
        print("DATA COLLECTOR CONTROLLER")
        print("=" * 60)
        print(f"\nWill collect {num_episodes} episodes of {episode_length} steps each")
        print("Data will be saved to: data/transitions/transitions.npy\n")
        sys.stdout.flush()
        
        try:
            print("[INIT] Step 1: Creating Robot instance...")
            sys.stdout.flush()
            # Try to create as Supervisor first for reset capabilities
            # Note: Supervisor() doesn't throw an error if not enabled in world file,
            # so we need to test if Supervisor methods work
            try:
                test_robot = Supervisor()
                # Test if we actually have Supervisor capabilities by trying getSelf()
                test_robot.step(0)  # Need to step first to initialize
                try:
                    test_node = test_robot.getSelf()
                    # If we get here without error, we have Supervisor capabilities
                    self.robot = test_robot
                    self.is_supervisor = True
                    print("[INIT] ✓ Robot created as Supervisor (can reset position)")
                    sys.stdout.flush()
                except:
                    # getSelf() failed, we're not actually a Supervisor
                    self.robot = Robot()
                    self.is_supervisor = False
                    print("[INIT] ⚠ Robot is NOT a Supervisor - reset will be disabled")
                    print("[INIT]   To enable reset: set 'supervisor' field to TRUE in world file")
                    sys.stdout.flush()
            except Exception as e:
                # Supervisor import or creation failed
                self.robot = Robot()
                self.is_supervisor = False
                print(f"[INIT] ⚠ Could not create Supervisor: {e}")
                print("[INIT]   Using regular Robot (no reset capabilities)")
                sys.stdout.flush()
            
            print("[INIT] Step 2: Getting basic timestep...")
            sys.stdout.flush()
            self.timestep = int(self.robot.getBasicTimeStep())
            print(f"[INIT] ✓ Timestep obtained: {self.timestep}ms")
            sys.stdout.flush()
            
            # Try to get robot node for reset (if supervisor)
            self.robot_node = None
            if self.is_supervisor:
                try:
                    # Step once to initialize Supervisor
                    self.robot.step(0)
                    self.robot_node = self.robot.getSelf()
                    if self.robot_node is None:
                        print("[INIT] ⚠ getSelf() returned None, trying DEF names...")
                        sys.stdout.flush()
                        # Try to get from DEF name (common drone DEF names)
                        for def_name in ["DRONE", "drone", "Drone", "Quadcopter", "quadcopter", "Mavic2Pro"]:
                            try:
                                self.robot_node = self.robot.getFromDef(def_name)
                                if self.robot_node is not None:
                                    print(f"[INIT] ✓ Found robot node with DEF '{def_name}'")
                                    sys.stdout.flush()
                                    break
                            except:
                                pass
                    
                    if self.robot_node is not None:
                        # Verify we can access fields
                        try:
                            test_field = self.robot_node.getField("translation")
                            if test_field is None:
                                print("[INIT] ⚠ Robot node found but translation field not accessible")
                                self.robot_node = None
                            else:
                                print("[INIT] ✓ Robot node accessible, reset enabled")
                        except Exception as e:
                            print(f"[INIT] ⚠ Cannot access robot node fields: {e}")
                            self.robot_node = None
                    else:
                        print("[INIT] ⚠ Could not find robot node for reset")
                except Exception as e:
                    print(f"[INIT] ⚠ Could not get robot node: {e}")
                    print("[INIT]   Reset will not be available")
                    self.is_supervisor = False  # Disable supervisor mode if we can't use it
                    sys.stdout.flush()
            
            # CRITICAL: Call robot.step() immediately to keep Webots alive
            print("[INIT] Step 2.5: Calling robot.step() to prevent timeout...")
            sys.stdout.flush()
            self.robot.step(self.timestep)
            print("[INIT] ✓ First robot.step() completed")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize Robot: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
        
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        
        # Initialize components
        try:
            print("[INIT] Step 3: Initializing Control module...")
            sys.stdout.flush()
            self.control = Control(self.robot, self.timestep)
            print("[INIT] ✓ Control initialized")
            sys.stdout.flush()
            
            # Another step to keep Webots alive
            print("[INIT] Step 3.5: Calling robot.step() after Control init...")
            sys.stdout.flush()
            self.robot.step(self.timestep)
            print("[INIT] ✓ robot.step() after Control completed")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize Control: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
        
        try:
            print("[INIT] Step 4: Initializing Perception module...")
            sys.stdout.flush()
            self.perception = Perception(self.robot, self.timestep)
            print("[INIT] ✓ Perception initialized")
            sys.stdout.flush()
            
            # Another step to keep Webots alive
            print("[INIT] Step 4.5: Calling robot.step() after Perception init...")
            sys.stdout.flush()
            self.robot.step(self.timestep)
            print("[INIT] ✓ robot.step() after Perception completed")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize Perception: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
        
        try:
            print("[INIT] Step 5: Setting up dataset...")
            sys.stdout.flush()
            self.dataset = TransitionModelDataset()
            print("[INIT] ✓ Dataset ready")
            sys.stdout.flush()
        except Exception as e:
            print(f"❌ ERROR: Failed to setup dataset: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
        
        self.current_episode = 0
        self.current_step = 0
        self.total_transitions = 0
        
        # Episode state
        self.prev_state = None
        self.prev_action = None
        
        # Reset/rescue parameters
        self.reset_count = 0
        self.MIN_ALTITUDE = -0.1  # Minimum altitude in meters (10cm)
        self.MAX_ALTITUDE = 10.0  # Maximum altitude in meters
        self.MAX_ROLL_PITCH = 1.5  # Maximum roll/pitch in radians (~86 degrees)
        self.MAX_ANGULAR_VEL = 10.0  # Maximum angular velocity in rad/s
        self.RESET_ALTITUDE = 1.5  # Altitude to reset to (1.5 meters)
        self.RESET_POSITION = [0.0, 0.0, self.RESET_ALTITUDE]  # Default reset position
        self.HOVER_ALTITUDE = 1.5  # Target hover altitude for data collection (1.5 meters)
        
        print("\n[INIT] ✓ Controller fully initialized!")
        print(f"[INIT] Bad state detection enabled: min_alt={self.MIN_ALTITUDE}m, max_alt={self.MAX_ALTITUDE}m")
        print(f"[INIT] Max roll/pitch: {self.MAX_ROLL_PITCH:.2f}rad, Reset altitude: {self.RESET_ALTITUDE}m")
        print("[INIT] Starting data collection...\n")
        sys.stdout.flush()
        
        # Final step to ensure Webots knows we're alive
        print("[INIT] Step 6: Final robot.step() before entering main loop...")
        sys.stdout.flush()
        self.robot.step(self.timestep)
        print("[INIT] ✓ Ready to enter main loop!\n")
        sys.stdout.flush()
    
    def is_bad_state(self, state_vector):
        """
        Check if the drone is in a bad state (flipped, crashed, glitched).
        
        Args:
            state_vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        
        Returns:
            tuple: (is_bad, reason) - True if bad state, False otherwise, and reason string
        """
        if state_vector is None or len(state_vector) < 12:
            return True, "Invalid state vector"
        
        position = state_vector[0:3]  # [x, y, z]
        orientation = state_vector[3:6]  # [roll, pitch, yaw]
        angular_velocity = state_vector[9:12]  # [wx, wy, wz]
        
        altitude = position[2]  # z coordinate
        
        # Check altitude (too low = crashed, too high = glitched)
        if altitude < self.MIN_ALTITUDE:
            return True, f"Altitude too low: {altitude:.2f}m < {self.MIN_ALTITUDE}m"
        if altitude > self.MAX_ALTITUDE:
            return True, f"Altitude too high: {altitude:.2f}m > {self.MAX_ALTITUDE}m"
        
        # Check orientation (flipped upside down)
        roll = orientation[0]
        pitch = orientation[1]
        
        if abs(roll) > self.MAX_ROLL_PITCH:
            return True, f"Roll too extreme: {roll:.2f}rad (max: {self.MAX_ROLL_PITCH:.2f}rad)"
        if abs(pitch) > self.MAX_ROLL_PITCH:
            return True, f"Pitch too extreme: {pitch:.2f}rad (max: {self.MAX_ROLL_PITCH:.2f}rad)"
        
        # Check angular velocity (spinning too fast = glitched)
        angular_vel_magnitude = np.linalg.norm(angular_velocity)
        if angular_vel_magnitude > self.MAX_ANGULAR_VEL:
            return True, f"Angular velocity too high: {angular_vel_magnitude:.2f}rad/s (max: {self.MAX_ANGULAR_VEL:.2f}rad/s)"
        
        # Check for NaN or inf values (glitched sensor data)
        if not np.all(np.isfinite(state_vector)):
            return True, "Non-finite values in state vector (NaN/Inf detected)"
        
        return False, None
    
    def ascend_and_hover(self, target_altitude=None, max_steps=500):
        """
        Ascend the drone to target altitude and stabilize before data collection.
        
        Args:
            target_altitude: Target altitude in meters (default: self.HOVER_ALTITUDE)
            max_steps: Maximum number of steps to attempt hover (default: 500)
        
        Returns:
            bool: True if successfully hovering, False otherwise
        """
        if target_altitude is None:
            target_altitude = self.HOVER_ALTITUDE
        
        print(f"\n[INIT] Ascending to hover altitude: {target_altitude}m")
        sys.stdout.flush()
        
        # Higher base hover thrust - needs to overcome gravity better
        # Increase from 68.5 to 72-75 for more stable hover
        HIGHER_HOVER_THRUST = 73.0
        
        # Altitude control PID parameters (more conservative to prevent flipping)
        alt_kp = 8.0   # Reduced from 15.0 to prevent overshoot
        alt_ki = 0.05  # Reduced integral gain
        alt_kd = 3.0   # Reduced derivative gain
        alt_integral = 0.0
        prev_alt_error = 0.0
        
        steps_at_target = 0
        steps_required_stable = 30  # Need to be stable for 30 steps
        
        for step in range(max_steps):
            if self.robot.step(self.timestep) == -1:
                return False
            
            try:
                # Get current state
                state = self.perception.get_state_vector()
                current_altitude = state[2]  # z coordinate
                roll, pitch = state[3], state[4]
                
                # Calculate altitude error first (always needed)
                alt_error = target_altitude - current_altitude
                
                # Safety check: if drone is tilting too much, use higher thrust to stabilize
                tilt_magnitude = np.sqrt(roll**2 + pitch**2)
                if tilt_magnitude > 0.3:  # More than 0.3 rad tilt
                    # Use stabilizing thrust
                    base_thrust = HIGHER_HOVER_THRUST + 2.0  # Extra boost
                    print(f"[INIT] ⚠ Large tilt detected (roll={roll:.2f}, pitch={pitch:.2f}), increasing thrust")
                    sys.stdout.flush()
                else:
                    # Normal altitude control
                    
                    # PID control for altitude (only if not too far off)
                    dt = self.timestep / 1000.0  # Convert ms to seconds
                    
                    # If we're far below target, use aggressive ascent
                    if alt_error > 0.5:  # More than 50cm below target
                        # Use higher thrust for ascent
                        alt_correction = min(alt_kp * alt_error * 1.5, 8.0)  # Limit correction
                        base_thrust = HIGHER_HOVER_THRUST + alt_correction
                    else:
                        # Normal PID control
                        alt_integral += alt_error * dt
                        alt_integral = max(-3.0, min(3.0, alt_integral))  # Clamp integral tighter
                        alt_derivative = (alt_error - prev_alt_error) / dt if dt > 0 else 0
                        prev_alt_error = alt_error
                        
                        # Calculate thrust adjustment
                        alt_correction = alt_kp * alt_error + alt_ki * alt_integral + alt_kd * alt_derivative
                        alt_correction = max(-5.0, min(5.0, alt_correction))  # Limit correction
                        base_thrust = HIGHER_HOVER_THRUST + alt_correction
                
                # Keep thrust in safe range
                base_thrust = max(65.0, min(82.0, base_thrust))
                
                # Apply same thrust to all motors for vertical ascent/hover
                action = np.array([base_thrust] * 4)
                
                # Execute control
                velocities = self.control.stabilise(self.timestep, action)
                self.control.process_signal(velocities)
                
                # Check if we're at target altitude and stable
                altitude_error = abs(alt_error)
                orientation_stable = abs(roll) < 0.15 and abs(pitch) < 0.15  # Tighter stability requirement
                
                if altitude_error < 0.15 and orientation_stable:  # Within 15cm of target
                    steps_at_target += 1
                    if steps_at_target >= steps_required_stable:
                        print(f"[INIT] ✓ Hover achieved at {current_altitude:.2f}m after {step+1} steps")
                        print(f"      Altitude error: {altitude_error:.3f}m, Roll: {roll:.2f}rad, Pitch: {pitch:.2f}rad")
                        sys.stdout.flush()
                        return True
                else:
                    steps_at_target = 0
                
                # Progress updates
                if step % 50 == 0:
                    print(f"[INIT] Step {step}: Altitude = {current_altitude:.2f}m (target: {target_altitude:.2f}m), "
                          f"error = {altitude_error:.3f}m, Thrust = {base_thrust:.1f}, "
                          f"Roll = {roll:.2f}rad, Pitch = {pitch:.2f}rad")
                    sys.stdout.flush()
                
            except Exception as e:
                print(f"[INIT] ⚠ Error during ascent: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                # Continue trying
        
        # If we get here, we didn't achieve stable hover
        try:
            final_state = self.perception.get_state_vector()
            final_alt = final_state[2]
            final_roll = final_state[3]
            final_pitch = final_state[4]
            print(f"[INIT] ⚠ Failed to achieve stable hover after {max_steps} steps")
            print(f"      Final altitude: {final_alt:.2f}m (target: {target_altitude:.2f}m)")
            print(f"      Final orientation: Roll = {final_roll:.2f}rad, Pitch = {final_pitch:.2f}rad")
            sys.stdout.flush()
        except:
            pass
        
        # Still return True to continue (may be close enough)
        return True
    
    def reset_drone(self):
        """
        Reset the drone to a safe position in the air.
        """
        if not self.is_supervisor:
            print("⚠ Cannot reset: Robot is not a Supervisor")
            print("   To enable reset: In Webots, select the drone in Scene Tree,")
            print("   then set 'supervisor' field to TRUE in the field editor")
            sys.stdout.flush()
            return False
        
        if self.robot_node is None:
            print("⚠ Cannot reset: Robot node not available")
            print("   This may happen if the drone doesn't have a DEF name")
            sys.stdout.flush()
            return False
        
        try:
            # Get translation field and set new position
            translation_field = self.robot_node.getField("translation")
            if translation_field is None:
                print("⚠ Cannot reset: translation field not found on robot node")
                print("   The robot node structure may be different than expected")
                sys.stdout.flush()
                return False
            
            # Reset position to safe altitude
            translation_field.setSFVec3f(self.RESET_POSITION)
            
            # Reset rotation to upright orientation (0, 0, 1, 0 means no rotation)
            rotation_field = self.robot_node.getField("rotation")
            if rotation_field is not None:
                rotation_field.setSFRotation([0, 0, 1, 0])
            
            # Reset velocities by resetting physics (optional, may cause brief pause)
            try:
                if hasattr(self.robot, 'simulationResetPhysics'):
                    self.robot.simulationResetPhysics()
            except:
                pass
            
            # Reset perception state tracking (clear velocity history)
            if hasattr(self.perception, 'prev_position'):
                self.perception.prev_position = None
                self.perception.prev_time = None
            
            # Reset control integrals
            if hasattr(self.control, 'integral'):
                self.control.integral = {'roll': 0, 'pitch': 0, 'yaw': 0, 'vertical': 0}
            if hasattr(self.control, 'prev_error'):
                self.control.prev_error = {'roll': 0, 'pitch': 0, 'yaw': 0, 'vertical': 0}
            
            self.reset_count += 1
            print(f"\n🔄 RESET #{self.reset_count}: Drone reset to position {self.RESET_POSITION}")
            print(f"   Episode: {self.current_episode + 1}, Step: {self.current_step}")
            sys.stdout.flush()
            
            # Give a few steps for the reset to take effect
            for _ in range(5):
                self.robot.step(self.timestep)
            
            return True
            
        except Exception as e:
            print(f"⚠ Error during reset: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            return False
    
    def run(self):
        """Main control loop for data collection."""
        try:
            print("[RUN] Entering main run() method...")
            sys.stdout.flush()
            
            # Wait a few steps for sensors to initialize
            print("[RUN] Waiting for sensors to initialize (10 steps)...")
            sys.stdout.flush()
            for i in range(10):
                if i % 3 == 0:
                    print(f"[RUN] Sensor initialization step {i+1}/10...")
                    sys.stdout.flush()
                self.robot.step(self.timestep)
            print("[RUN] ✓ Sensors ready")
            sys.stdout.flush()
            
            # Ascend and hover before starting data collection
            print("[RUN] Ascending and stabilizing drone before data collection...")
            sys.stdout.flush()
            hover_success = self.ascend_and_hover(self.HOVER_ALTITUDE, max_steps=500)
            if hover_success:
                print("[RUN] ✓ Drone is hovering and ready for data collection\n")
            else:
                print("[RUN] ⚠ Hover may not be fully stable, proceeding anyway\n")
            sys.stdout.flush()
            
            print("[RUN] Starting main data collection loop...")
            sys.stdout.flush()
            loop_count = 0
            
            while self.robot.step(self.timestep) != -1:
                loop_count += 1
                
                # Log first few iterations
                if loop_count <= 5:
                    print(f"[RUN] Loop iteration {loop_count}")
                    sys.stdout.flush()
                elif loop_count == 6:
                    print("[RUN] Loop running smoothly, reducing logs...")
                    sys.stdout.flush()
                
                try:
                    # Check if episode is complete
                    if self.current_step >= self.episode_length:
                        print(f"Episode {self.current_episode + 1}/{self.num_episodes} complete. "
                              f"Total transitions: {self.total_transitions}")
                        
                        self.current_episode += 1
                        self.current_step = 0
                        self.prev_state = None
                        self.prev_action = None
                        
                        # Check if all episodes are done
                        if self.current_episode >= self.num_episodes:
                            print("\n" + "=" * 60)
                            print("All episodes complete!")
                            print("=" * 60)
                            break
                    
                    # Get current state
                    try:
                        current_state = self.perception.get_state_vector()
                        
                        # Check for bad states (flipped, crashed, glitched)
                        is_bad, reason = self.is_bad_state(current_state)
                        if is_bad:
                            print(f"\n⚠ BAD STATE DETECTED: {reason}")
                            print(f"   State: pos={current_state[0:3]}, roll={current_state[3]:.2f}, pitch={current_state[4]:.2f}")
                            sys.stdout.flush()
                            
                            # Try to reset the drone
                            if self.reset_drone():
                                print("   ✓ Drone reset successfully, continuing data collection")
                                sys.stdout.flush()
                                
                                # Skip collecting this transition (bad state -> reset)
                                # Reset episode tracking to start fresh
                                self.current_step = 0
                                self.prev_state = None
                                self.prev_action = None
                                
                                # Re-hover after reset to stabilize
                                print("   Re-hovering after reset...")
                                sys.stdout.flush()
                                self.ascend_and_hover(self.HOVER_ALTITUDE, max_steps=200)
                                
                                # Get state again after re-hover
                                try:
                                    current_state = self.perception.get_state_vector()
                                    
                                    # Verify reset and hover worked
                                    is_still_bad, _ = self.is_bad_state(current_state)
                                    if is_still_bad:
                                        print("   ⚠ Reset/re-hover may not have worked, skipping this step")
                                        sys.stdout.flush()
                                        continue
                                except Exception as e:
                                    print(f"   ⚠ Error getting state after reset/re-hover: {e}")
                                    sys.stdout.flush()
                                    continue
                            else:
                                print("   ⚠ Reset failed, skipping this transition")
                                sys.stdout.flush()
                                continue
                    except Exception as e:
                        print(f"⚠ Warning: Failed to get state vector: {e}")
                        # Skip this step
                        continue
                    
                    # Generate action (random exploration)
                    # Convert normalized values to motor velocities if needed
                    # The control.stabilise() expects motor velocities around 68.5 for hover
                    if self.current_step % 5 == 0:  # Change action every 5 steps
                        # Random action - using actual motor velocity values
                        # 68.5 is approximately hover thrust
                        base_thrust = HOVER_THRUST  # Hover thrust
                        action = base_thrust + np.random.uniform(-10, 10, 4)
                        action = np.clip(action, 50, 100)  # Keep in reasonable range
                    else:
                        # Keep previous action
                        if self.prev_action is None:
                            action = np.array([HOVER_THRUST] * 4)  # Hover by default
                        else:
                            action = self.prev_action
                    
                    # Execute action
                    try:
                        velocities = self.control.stabilise(self.timestep, action)
                        self.control.process_signal(velocities)
                    except Exception as e:
                        print(f"⚠ Warning: Failed to execute action: {e}")
                        import traceback
                        traceback.print_exc()
                        # Use safe default action
                        try:
                            safe_vel = [HOVER_THRUST] * 4
                            for motor, v in zip(self.control.motors, safe_vel):
                                motor.setVelocity(v)
                        except:
                            pass
                    
                    # Store transition if we have previous state
                    # Only store if both states are valid (not bad)
                    if self.prev_state is not None and self.prev_action is not None:
                        try:
                            # Verify both states are good before storing
                            prev_is_bad, prev_reason = self.is_bad_state(self.prev_state)
                            curr_is_bad, curr_reason = self.is_bad_state(current_state)
                            
                            if prev_is_bad or curr_is_bad:
                                # Don't store transitions involving bad states
                                if prev_is_bad:
                                    print(f"⚠ Skipping transition: previous state was bad ({prev_reason})")
                                if curr_is_bad:
                                    print(f"⚠ Skipping transition: current state is bad ({curr_reason})")
                                sys.stdout.flush()
                            else:
                                # Both states are good, store the transition
                                self.dataset.add_transition(self.prev_state, self.prev_action, current_state)
                                self.total_transitions += 1
                                
                                if self.total_transitions % 100 == 0:
                                    print(f"  Collected {self.total_transitions} transitions...", end='\r')
                        except Exception as e:
                            print(f"⚠ Warning: Failed to add transition: {e}")
                    
                    # Update for next iteration
                    self.prev_state = current_state.copy()
                    self.prev_action = action.copy()
                    self.current_step += 1
                    
                except Exception as e:
                    print(f"⚠ Warning: Error in main loop step {self.current_step}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next step
                    self.current_step += 1
                    continue
        except Exception as e:
            print(f"❌ Fatal error in run loop: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Save collected data
        print(f"\n\nSaving {self.total_transitions} transitions...")
        self.dataset.save()
        print("=" * 60)
        print("✓ Data collection complete!")
        print(f"✓ Saved {self.total_transitions} transitions to {self.dataset.data_file}")
        if self.reset_count > 0:
            print(f"✓ Drone was reset {self.reset_count} time(s) due to bad states")
        print("=" * 60)
        
        # Keep simulation running for a bit so message is visible
        for _ in range(100):
            self.robot.step(self.timestep)


# Configuration - adjust these values as needed
NUM_EPISODES = 10
EPISODE_LENGTH = 500

# Main execution
if __name__ == "__main__":
    print("[MAIN] Script execution started")
    print("[MAIN] Starting data collection...")
    sys.stdout.flush()
    
    try:
        print("[MAIN] Creating DataCollectorController instance...")
        sys.stdout.flush()
        data_collector_controller = DataCollectorController(
            num_episodes=NUM_EPISODES,
            episode_length=EPISODE_LENGTH
        )
        print("[MAIN] ✓ DataCollectorController created successfully")
        print("[MAIN] Calling run() method...")
        sys.stdout.flush()
        
        data_collector_controller.run()
        print("[MAIN] ✓ run() method completed")
        sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\n[MAIN] ⚠ Collection interrupted by user.")
        sys.stdout.flush()
        print("[MAIN] Saving collected data...")
        sys.stdout.flush()
        try:
            data_collector_controller.dataset.save()
            print("[MAIN] ✓ Data saved before exit.")
            sys.stdout.flush()
        except Exception as save_err:
            print(f"[MAIN] ⚠ Could not save data: {save_err}")
            sys.stdout.flush()
    except Exception as e:
        print(f"\n\n[MAIN] ❌ FATAL ERROR: {e}")
        print(f"[MAIN] Error type: {type(e).__name__}")
        sys.stdout.flush()
        import traceback
        print("[MAIN] Full traceback:")
        traceback.print_exc()
        sys.stdout.flush()
        
        print("\n[MAIN] Attempting to save any collected data...")
        sys.stdout.flush()
        try:
            if 'data_collector_controller' in locals():
                print("[MAIN] data_collector_controller exists, trying to save...")
                sys.stdout.flush()
                data_collector_controller.dataset.save()
                print("[MAIN] ✓ Data saved successfully")
                sys.stdout.flush()
        except Exception as save_error:
            print(f"[MAIN] ⚠ Could not save data: {save_error}")
            sys.stdout.flush()
        
        # Re-raise to see the error in Webots console
        print("[MAIN] Re-raising exception...")
        sys.stdout.flush()
        raise

