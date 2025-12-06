import torch
import torch.nn as nn
import numpy as np
from tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.model_based import ModelBasedEnvBase
from tensordict.nn import TensorDictModule
from torchrl.modules import ValueOperator, MLP, WorldModelWrapper
from torchrl.objectives.value import TDLambdaEstimator
from torchrl.planners.mppi import MPPIPlanner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DroneModelBasedEnv(ModelBasedEnvBase):
    """
    Model-based environment for drone path planning.
    State: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    Action: [motor1_vel, motor2_vel, motor3_vel, motor4_vel]
    """
    def __init__(self, world_model, device="cpu", dtype=None, batch_size=None):
        super().__init__(world_model, device=device, dtype=dtype, batch_size=batch_size)
        
        # position (3) + orientation (3) + linear velocity (3) + angular velocity (3) = 12
        state_dim = 12
        # 4 motor velocities
        action_dim = 4
        
        self.state_spec = CompositeSpec(
            state=UnboundedContinuousTensorSpec((state_dim,))
        )

        self.observation_spec = CompositeSpec(
            state=UnboundedContinuousTensorSpec((state_dim,))
        )

        self.action_spec = UnboundedContinuousTensorSpec((action_dim,))
        self.reward_spec = UnboundedContinuousTensorSpec((1,))
    
    def _reset(self, tensordict: TensorDict) -> TensorDict:
        """
        Reset the environment to an initial state.
        """
        if tensordict is None or tensordict.is_empty():
            tensordict = TensorDict(
                {},
                batch_size=self.batch_size,
                device=self.device,
            )
        
        # Initialise state: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        # Start at origin with zero velocities
        initial_state = torch.zeros(
            (*self.batch_size, 12),
            device=self.device,
            dtype=self.dtype
        )
        
        tensordict = tensordict.update(
            self.state_spec.zero()
        )

        tensordict["state"] = initial_state

        tensordict = tensordict.update(
            self.observation_spec.zero()
        )

        tensordict["state"] = initial_state
        
        return tensordict


class PathPlanner():
    def __init__(self, robot, timestep, perception=None, goal_positions=None, world_bounds=None):
        """
        Initialize MPPI path planner with world environment awareness.
        
        Args:
            robot: Webots robot instance
            timestep: Simulation timestep
            perception: Perception module instance for obstacle detection
            goal_positions: List of target positions [[x1, y1, z1], [x2, y2, z2], ...]
            world_bounds: Dict with 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
        """
        self.robot = robot
        self.timestep = timestep
        self.perception = perception
        self.goal_positions = goal_positions if goal_positions else []
        
        # World boundaries for collision checking
        self.world_bounds = world_bounds if world_bounds else {
            'x_min': -20, 'x_max': 20,
            'y_min': -20, 'y_max': 20,
            'z_min': 0.1, 'z_max': 10.0  # z_min > 0 to avoid ground collision
        }
        
        state_dim = 12
        action_dim = 4
        
        # predicts next state from current state and action
        transition_model = TensorDictModule(
            MLP(
                in_features=state_dim + action_dim,
                out_features=state_dim,
                activation_class=nn.ReLU,
                activate_last_layer=False,
                depth=2,
                num_cells=64,
            ),
            in_keys=["state", "action"],
            out_keys=["state"],
        )
        

        reward_model = TensorDictModule(
            MLP(
                in_features=state_dim,
                out_features=1,
                activation_class=nn.ReLU,
                activate_last_layer=False,
                depth=2,
                num_cells=32,
            ),
            in_keys=["state"],
            out_keys=["reward"],
        )
        
        # include world information into reward model
        world_aware_reward = WorldAwareRewardModel(
            reward_model,
            perception=self.perception,
            goal_positions=self.goal_positions,
            world_bounds=self.world_bounds,
            device=str(device)
        )
        
        world_model = WorldModelWrapper(
            transition_model,
            world_aware_reward,
        )
        
        self.env = DroneModelBasedEnv(
            world_model,
            device=str(device),
            dtype=torch.float32,
            batch_size=[],
        )
        
        # value network for advantage estimation
        value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        value_net = ValueOperator(value_net, in_keys=["state"])
        
        # advantage estimator
        self.adv = TDLambdaEstimator(
            gamma=0.99,
            lmbda=0.95,
            value_network=value_net,
        )
        
        self.planner = MPPIPlanner(
            self.env,
            self.adv,
            temperature=1.0,
            planning_horizon=10,
            optim_steps=11,
            num_candidates=7,
            top_k=3,
        )
    
    def update_goals(self, new_goal_positions):
        """Update goal positions (e.g., when targets are found)."""
        self.goal_positions = new_goal_positions

        if hasattr(self.env, "world_model"):
            if hasattr(self.env.world_model, "reward_model"):
                if hasattr(self.env.world_model.reward_model, "goal_positions"):
                    self.env.world_model.reward_model.goal_positions = new_goal_positions


class WorldAwareRewardModel(TensorDictModule):
    """
    Reward model that incorporates world environment information:
    - Collision penalties
    - Goal rewards
    - Boundary penalties
    - Stability rewards
    """
    def __init__(self, base_reward_model, perception=None, goal_positions=None, 
                 world_bounds=None, device="cpu"):
        # get base reward model if it has one otherwise use default
        if hasattr(base_reward_model, "module"):
            module = base_reward_model.module
        else:
            module = base_reward_model
        
        super().__init__(
            module,
            in_keys=base_reward_model.in_keys,
            out_keys=base_reward_model.out_keys
        )

        self.base_reward_model = base_reward_model
        self.perception = perception
        self.goal_positions = goal_positions if goal_positions else []
        self.world_bounds = world_bounds if world_bounds else {}
        self.device = device
        
        # Reward weights
        self.collision_penalty = -100.0
        self.boundary_penalty = -50.0
        self.goal_reward = 10.0
        self.goal_distance_weight = -0.1
        self.stability_reward = 1.0
        self.min_safe_distance = 0.5
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """
        Compute reward with world awareness.
        State format: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        """
        # Get base reward from neural network
        tensordict = self.base_reward_model(tensordict)
        base_reward = tensordict["reward"]
        
        # get state tensor [batch_size, 12]
        state = tensordict["state"]
        
        # handle batched states
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            is_single = True
        else:
            is_single = False
        
        batch_size = state.shape[0]
        rewards = base_reward.clone() if base_reward.shape[0] == batch_size else base_reward.repeat(batch_size, 1)
        
        # Convert to numpy for easier processing
        state_np = state.detach().cpu().numpy()
        
        for i in range(batch_size):
            pos = state_np[i, :3]  # [x, y, z]
            orientation = state_np[i, 3:6]  # [roll, pitch, yaw]
            velocity = state_np[i, 6:9]  # [vx, vy, vz]
            angular_vel = state_np[i, 9:12]  # [wx, wy, wz]
            
            # check collisions with world boundaries
            if self._check_boundary_collision(pos):
                rewards[i] += self.boundary_penalty
            
            # check collisions with obstacles
            if self.perception:
                if self._check_obstacle_collision(pos):
                    rewards[i] += self.collision_penalty
            
            # reward proximity to goals
            if len(self.goal_positions) > 0:
                min_goal_distance = float("inf")

                for goal in self.goal_positions:
                    goal_tensor = torch.tensor(goal, device=self.device)
                    pos_tensor = torch.tensor(pos, device=self.device)
                    distance = torch.norm(pos_tensor - goal_tensor).item()
                    min_goal_distance = min(min_goal_distance, distance)
                
                # reward getting closer to goals
                rewards[i] += self.goal_distance_weight * min_goal_distance
                
                # large reward for reaching goal (within 0.5m)
                if min_goal_distance < 0.5:
                    rewards[i] += self.goal_reward
            
            # reward being stable
            angular_vel_magnitude = np.linalg.norm(angular_vel)

            if angular_vel_magnitude < 0.5:
                rewards[i] += self.stability_reward
            else:
                # penalise being unstable
                rewards[i] -= 0.5 * angular_vel_magnitude
            
            # penalise being too low or high
            if pos[2] < self.world_bounds.get("z_min", 0.5):
                rewards[i] -= 20.0
            elif pos[2] > self.world_bounds.get("z_max", 5.0):
                rewards[i] -= 10.0
        
        if is_single:
            rewards = rewards.squeeze(0)
        
        tensordict["reward"] = rewards

        return tensordict
    
    def _check_boundary_collision(self, position):
        """
        check if position is outside world boundaries
        """

        if not self.world_bounds:
            return False
        
        x, y, z = position[0], position[1], position[2]
        
        # check if position is outside of bounds
        if (x < self.world_bounds.get("x_min", -float("inf")) or
            x > self.world_bounds.get("x_max", float("inf")) or
            y < self.world_bounds.get("y_min", -float("inf")) or
            y > self.world_bounds.get("y_max", float("inf")) or
            z < self.world_bounds.get("z_min", 0) or
            z > self.world_bounds.get("z_max", float("inf"))):

            return True

        # return false if in bounds
        return False
    
    def _check_obstacle_collision(self, position):
        """
        Check if position would collide with obstacles
        """
        if not self.perception:
            return False
        
        distances = self.perception.get_obstacle_distances()

        # check if obstacles are too close
        if distances:
            for dist in distances.values():
                if dist < self.min_safe_distance:
                    return True

        return False