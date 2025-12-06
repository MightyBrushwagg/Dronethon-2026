import torch
import torch.nn as nn
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
        
        # State: position (3) + orientation (3) + linear velocity (3) + angular velocity (3) = 12
        state_dim = 12
        # Action: 4 motor velocities
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
    def __init__(self, robot, timestep):
        self.robot = robot
        self.timestep = timestep
        
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
        
        # predicts reward from state
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
        
        world_model = WorldModelWrapper(
            transition_model,
            reward_model,
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