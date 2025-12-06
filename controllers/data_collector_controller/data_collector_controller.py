"""
Webots controller for collecting training data.

This controller should be set as the drone's controller in Webots.
It will collect state-action-next_state transitions and save them.

To use:
1. Open Webots and load a world file (e.g., worlds/sample-map-1.wbt)
2. In the Scene Tree, select your drone
3. In the drone's properties, set the 'controller' field to: 'data_collector_controller'
4. (Optional) Edit NUM_EPISODES and EPISODE_LENGTH below to adjust collection
5. Click Play in Webots to start the simulation
6. The controller will automatically collect data and save it when done
7. Data will be saved to: controllers/data_collector_controller/data/transitions/transitions.npy

After collection, train the model with:
    cd controllers/drone_controller
    python -m train_transition_model --mode train --epochs 100

Or create a training script in the drone_controller folder.
"""

import warnings
import os
# Suppress numpy warnings early
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import sys
from pathlib import Path

# Add parent directory to path to import from drone_controller
controller_dir = Path(__file__).parent
drone_controller_dir = controller_dir.parent / "drone_controller"

# Ensure the path is absolute
if not drone_controller_dir.is_absolute():
    # Get absolute path
    project_root = controller_dir.parent.parent  # Go up to controllers, then project root
    drone_controller_dir = project_root / "controllers" / "drone_controller"

sys.path.insert(0, str(drone_controller_dir))

# Import Webots and custom modules with error handling
try:
    from controller import Robot
except ImportError as e:
    print(f"ERROR: Could not import Robot from controller: {e}")
    raise

try:
    from control import Control
except ImportError as e:
    print(f"ERROR: Could not import Control: {e}")
    print(f"Looking for control.py in: {drone_controller_dir}")
    raise

try:
    from perception import Perception
except ImportError as e:
    print(f"ERROR: Could not import Perception: {e}")
    print(f"Looking for perception.py in: {drone_controller_dir}")
    raise

import numpy as np


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
        
        try:
            print("Initializing Robot...")
            self.robot = Robot()
            self.timestep = int(self.robot.getBasicTimeStep())
            print(f"✓ Robot initialized, timestep: {self.timestep}ms")
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize Robot: {e}")
            raise
        
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        
        # Initialize components
        try:
            print("Initializing Control module...")
            self.control = Control(self.robot, self.timestep)
            print("✓ Control initialized")
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize Control: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        try:
            print("Initializing Perception module...")
            self.perception = Perception(self.robot, self.timestep)
            print("✓ Perception initialized")
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize Perception: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        try:
            print("Setting up dataset...")
            self.dataset = TransitionModelDataset()
            print("✓ Dataset ready")
        except Exception as e:
            print(f"❌ ERROR: Failed to setup dataset: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        self.current_episode = 0
        self.current_step = 0
        self.total_transitions = 0
        
        # Episode state
        self.prev_state = None
        self.prev_action = None
        
        print("\n✓ Controller initialized")
        print("Starting data collection...\n")
    
    def run(self):
        """Main control loop for data collection."""
        while self.robot.step(self.timestep) != -1:
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
            current_state = self.perception.get_state_vector()
            
            # Generate action (random exploration)
            if self.current_step % 5 == 0:  # Change action every 5 steps
                # Random action with some structure
                base_thrust = 1.0  # Hover thrust
                action = base_thrust + np.random.uniform(-0.5, 0.5, 4)
                action = np.clip(action, 0.5, 1.5)  # Keep in reasonable range
            else:
                # Keep previous action
                if self.prev_action is None:
                    action = np.array([1.0] * 4)
                else:
                    action = self.prev_action
            
            # Execute action
            velocities = self.control.stabilise(self.timestep, action)
            self.control.process_signal(velocities)
            
            # Store transition if we have previous state
            if self.prev_state is not None and self.prev_action is not None:
                self.dataset.add_transition(self.prev_state, self.prev_action, current_state)
                self.total_transitions += 1
                
                if self.total_transitions % 100 == 0:
                    print(f"  Collected {self.total_transitions} transitions...", end='\r')
            
            # Update for next iteration
            self.prev_state = current_state.copy()
            self.prev_action = action.copy()
            self.current_step += 1
        
        # Save collected data
        print(f"\n\nSaving {self.total_transitions} transitions...")
        self.dataset.save()
        print("=" * 60)
        print("✓ Data collection complete!")
        print(f"✓ Saved {self.total_transitions} transitions to {self.dataset.data_file}")
        print("=" * 60)
        
        # Keep simulation running for a bit so message is visible
        for _ in range(100):
            self.robot.step(self.timestep)


# Configuration - adjust these values as needed
NUM_EPISODES = 10
EPISODE_LENGTH = 500

# Main execution
if __name__ == "__main__":
    print("Starting data collection...")
    
    try:
        data_collector_controller = DataCollectorController(
            num_episodes=NUM_EPISODES,
            episode_length=EPISODE_LENGTH
        )
        data_collector_controller.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Collection interrupted by user.")
        print("Saving collected data...")
        try:
            data_collector_controller.dataset.save()
            print("✓ Data saved before exit.")
        except:
            print("⚠ Could not save data.")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving any collected data...")
        try:
            if 'data_collector_controller' in locals():
                data_collector_controller.dataset.save()
        except Exception as save_error:
            print(f"⚠ Could not save data: {save_error}")
        # Re-raise to see the error in Webots console
        raise

