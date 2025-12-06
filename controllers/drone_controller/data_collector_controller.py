"""
Webots controller for collecting training data.

This controller should be set as the drone's controller in Webots.
It will collect state-action-next_state transitions and save them.

To use:
1. Open Webots
2. Load a world file with a drone
3. In the Scene Tree, select the drone
4. Set the 'controller' field to 'data_collector_controller'
5. Run the simulation
6. The controller will collect data for the specified number of episodes
7. Data will be saved to data/transitions/transitions.npy
"""

from controller import Robot
from control import Control
from perception import Perception
from train_transition_model import TransitionModelDataset
import numpy as np
import sys


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
        
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        
        # Initialize components
        self.control = Control(self.robot, self.timestep)
        self.perception = Perception(self.robot, self.timestep)
        self.dataset = TransitionModelDataset()
        
        self.current_episode = 0
        self.current_step = 0
        self.total_transitions = 0
        
        # Episode state
        self.prev_state = None
        self.prev_action = None
        
        print("✓ Controller initialized")
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
                # Motor velocities typically range from ~50 to ~100
                # 68.5 is approximately the hover thrust
                base_thrust = 68.5  # Hover thrust
                action = base_thrust + np.random.uniform(-10, 10, 4)
                action = np.clip(action, 50, 100)  # Keep in reasonable range
            else:
                # Keep previous action
                if self.prev_action is None:
                    action = np.array([68.5] * 4)
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

# Create and run controller
if __name__ == "__main__":
    try:
        controller = DataCollectorController(
            num_episodes=NUM_EPISODES,
            episode_length=EPISODE_LENGTH
        )
        controller.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Collection interrupted by user.")
        print("Saving collected data...")
        controller.dataset.save()
        print("✓ Data saved before exit.")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving any collected data...")
        try:
            controller.dataset.save()
        except:
            pass

