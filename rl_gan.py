from copy import deepcopy
import gym
import drawing_env
from drawing_env.envs.ops import *
import numpy as np
import tensorflow as tf

import tensorforce
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.execution import Runner
import tforce_additions.preprocessors

#BETA1 = 0.5
LEARNING_RATE = 1e-3
#BATCH_SIZE = 1
DIMENSION = 3
NUM_EPISODES = 50000
NUM_POSSIBLE_PIXEL_COLORS = 2
EPSILON_GREEDY_START = 0.4
EPSILON_GREEDY_PER_EPISODE_DECAY = 0.9999

env_tforce = OpenAIGym('DrawEnv-v0') # The Gym environment is accessible with env_tforce.gym

# General environment info
num_pixels = DIMENSION*DIMENSION
num_actions_per_pixel = NUM_POSSIBLE_PIXEL_COLORS
discount = drawing_env.envs.draw_env.GAMMA

# Generic agent specs for TensorForce

# Note: TensorForce attempts to access MultiDiscrete's gym.observation_space field
# which is not supported as of gym 0.10.x; the following line is a workaround
states = dict(shape=env_tforce.gym.observation_space.nvec.size, type='int')
actions = env_tforce.actions
adam_optimizer = dict(type='adam', learning_rate=LEARNING_RATE)
tensorboard_summary_spec_generic = dict(steps=10, # log TensorBoard info every 10 steps
                                        labels=['losses', 'network_variables', 'inputs', 'gradients_scalar'])

# DQN agent
from tensorforce.agents import DQNAgent

# Architecture: see tensorforce/core/networks/layer.py for documentation
# and tensorforce/examples/configs for example setups
dqn_spec = [
        # Layer 0: Cast state to tf.float32 so tensorforce doesn't crash when matmul with float32 weights
        dict(type='tforce_additions.layers.Cast', dtype=tf.float32),
        
        # Layer 1: fully connected, num states->num actions, LReLU activation
        dict(type='linear', size=num_pixels*num_actions_per_pixel),
        dict(type='nonlinearity', name='lrelu', alpha=0.2), # alpha: slope at x < 0
        
        # Layer 2: fully connected, num actions->num actions, LReLU activation
        dict(type='linear', size=num_pixels*num_actions_per_pixel),
        dict(type='nonlinearity', name='lrelu', alpha=0.2), 
]

#dqn_summary_spec = deepcopy(tensorboard_summary_spec_generic)
#dqn_summary_spec['directory'] = 'results/dqn/'

# TODO: investigate actions_exploration parameter. Passed to Model superclass and to Exploration
# object to initialize an exploration strategy, not sure what exploration types are valid/what params
# NOTE: Arguments don't appear to allow configuring loss function other than huber_loss;
#       assuming uses MSVE of Q-value as in Mnih et al. 2015
dqn_agent = DQNAgent(states, actions, dqn_spec, discount=discount,
                     optimizer=adam_optimizer,
                     memory=None, # Experience replay buffer; see tensorforce doc for default vals
                     target_sync_frequency=10000, # How often to update the target network
                     target_update_weight=1.0, # Has something to do with updating target network? Default val
#                     summarizer=dqn_summary_spec # TODO: causes weird issue where dqn/global_timestep depends on state which isn't fed
)

runner = Runner(dqn_agent, env_tforce)
def ep_finished(runnr):
        print(runner.episode)
        # Notify environment to use current runner policy as rollout policy for reward estimation
#        runnr.environment.gym.rollout_policy = (lambda s: runnr.agent.act(s)) # TODO this is bugged, causes ScatterUpdate errors
        if runnr.episode % 10 == 0:
                print("Finished episode {ep} after {ts} timesteps".format(ep=runnr.episode + 1, ts=runnr.timestep + 1))
                print("Episode reward: {}".format(runnr.episode_rewards[-1]))
                print("Average of last 10 rewards: {}".format(np.mean(runnr.episode_rewards[-10:])))
        return True

# TODO: Investigate how long to set max_episode_timesteps
runner.run(num_episodes = 100, max_episode_timesteps=100, episode_finished=ep_finished)
