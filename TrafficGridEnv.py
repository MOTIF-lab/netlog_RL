import gym
from gym import spaces
import numpy as np
import pynetlogo
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetLogoTrafficGridEnv(gym.Env):
    """Custom Environment for NetLogo Traffic Grid Model"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(NetLogoTrafficGridEnv, self).__init__()

        # Initialize NetLogoLink
        self.netlogo_home = '/home/henry/Desktop/NetLogo 6.2.2/'
        self.netlogo_version = '6.2.2'

        # Verify NetLogo home path
        if not os.path.exists(self.netlogo_home):
            raise FileNotFoundError(f"NetLogo home directory not found at {self.netlogo_home}")

        # Initialize the NetLogo link
        self.netlogo = pynetlogo.NetLogoLink(gui=False, netlogo_home=self.netlogo_home)

        # Define the path to your NetLogo model
        self.model_path = os.path.join(self.netlogo_home, 'app/models/Sample Models/Social Science/Traffic Grid.nlogo')

        # Verify model path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"NetLogo model not found at {self.model_path}")

        # Load the NetLogo model
        self.netlogo.load_model(self.model_path)

        # Set the random seed for reproducibility
        random_seed = 42
        self.netlogo.command(f'random-seed {random_seed}')
        np.random.seed(random_seed)

        # Get the grid size
        self.grid_size_x = int(self.netlogo.report('grid-size-x'))
        self.grid_size_y = int(self.netlogo.report('grid-size-y'))
        self.num_intersections = self.grid_size_x * self.grid_size_y

        # Define action and observation space
        self.action_space = spaces.MultiBinary(self.num_intersections)
        low = np.zeros((self.num_intersections, 3), dtype=int)
        high = np.ones((self.num_intersections, 3), dtype=int)
        high[:, 1:] = 10  # Assuming max 10 cars per direction
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.current_tick = 0
        self.max_ticks = 1000  # Maximum simulation steps per episode

        # Custom setup
        self._setup_simulation()

    def _setup_simulation(self):
        try:
            self.netlogo.command('setup')
        except Exception as e:
            logger.error(f"Error during setup: {str(e)}")
            raise

    def reset(self):
        self._setup_simulation()
        obs = self._get_observation()
        self.current_tick = 0
        return obs

    def step(self, action):
        # Execute the action for each intersection
        for i in range(self.num_intersections):
            if action[i] == 1:
                self._switch_light(i)

        # Advance the simulation by one tick
        try:
            self.netlogo.command('go')
        except Exception as e:
            logger.error(f"Error during 'go' command: {str(e)}")
            raise

        self.current_tick += 1

        # Get the new observation
        obs = self._get_observation()

        # Compute reward (negative of total waiting time)
        try:
            total_wait_time = int(self.netlogo.report('sum [wait-time] of turtles'))
            reward = -total_wait_time
        except Exception as e:
            logger.error(f"Error computing reward: {str(e)}")
            reward = 0

        # Check if the episode is done
        done = self.current_tick >= self.max_ticks

        # Additional info
        info = {}
        try:
            info['num_cars_stopped'] = int(self.netlogo.report('num-cars-stopped'))
            info['avg_speed'] = float(self.netlogo.report('mean [speed] of turtles'))
        except Exception as e:
            logger.error(f"Error getting additional info: {str(e)}")

        return obs, reward, done, info

    def render(self, mode='human'):
        # Export the view to an image file
        export_path = os.path.join(os.getcwd(), 'simulation_state.png')
        try:
            self.netlogo.command(f'export-view "{export_path}"')
        except Exception as e:
            logger.error(f"Error exporting view: {str(e)}")
            return None

        # Read the exported image
        img = plt.imread(export_path)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            plt.imshow(img)
            plt.axis('off')
            plt.savefig('rendered_state.png')
            logger.info("Rendered state saved as 'rendered_state.png'")
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        self.netlogo.kill_workspace()

    def _get_observation(self):
        obs = []
        for y in range(self.grid_size_y):
            for x in range(self.grid_size_x):
                try:
                    green_light_up = int(self.netlogo.report(f'[green-light-up?] of one-of intersections with [my-row = {y} and my-column = {x}]'))
                    cars_waiting_ns = int(self.netlogo.report(f'count turtles-on patches with [pxcor = {x} * grid-x-inc - floor(grid-x-inc - 1) and pycor = {y} * grid-y-inc and pcolor = red]'))
                    cars_waiting_ew = int(self.netlogo.report(f'count turtles-on patches with [pycor = {y} * grid-y-inc and pxcor = {x} * grid-x-inc - floor(grid-x-inc - 1) and pcolor = red]'))
                    obs.append([green_light_up, cars_waiting_ns, cars_waiting_ew])
                except Exception as e:
                    logger.error(f"Error getting observation for intersection ({x}, {y}): {str(e)}")
                    obs.append([0, 0, 0])  # Default values in case of error
        return np.array(obs, dtype=np.int32)

    def _switch_light(self, intersection_id):
        y = intersection_id // self.grid_size_x
        x = intersection_id % self.grid_size_x
        try:
            self.netlogo.command(f'ask one-of intersections with [my-row = {y} and my-column = {x}] [ set green-light-up? not green-light-up? set-signal-colors ]')
        except Exception as e:
            logger.error(f"Error switching light at intersection ({x}, {y}): {str(e)}")

def test_environment():
    # Create an instance of the environment
    env = NetLogoTrafficGridEnv()

    # Reset the environment and get the initial observation
    obs = env.reset()
    logger.info("Initial Observation:")
    logger.info(obs)
    logger.info(f"Observation shape: {obs.shape}")

    # Run a few steps with random actions
    for i in range(5):
        logger.info(f"\nStep {i + 1}")

        # Generate a random action
        action = env.action_space.sample()
        logger.info(f"Action: {action}")

        # Take a step in the environment
        obs, reward, done, info = env.step(action)

        logger.info(f"Observation: {obs}")
        logger.info(f"Reward: {reward}")
        logger.info(f"Done: {done}")
        logger.info(f"Info: {info}")

    # Render the final state
    env.render(mode='human')

    # Close the environment
    env.close()

if __name__ == "__main__":
    test_environment()