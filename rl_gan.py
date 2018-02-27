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
#actions = dict(num_actions=env_tforce.gym.action_space.n, type='int') # Passed as ints so compatible with TensorBoard histograms
adam_optimizer = dict(type='adam', learning_rate=LEARNING_RATE)
tensorboard_summary_spec_generic = dict(steps=10, # log TensorBoard info every 10 steps
                                        labels=['losses', 'network_variables', 'inputs', 'gradients_scalar'])

# Preprocessing: cast int-tuple states to float-tuples because tensorforce layers
# expect float32 inputs
preprocessing_config = [dict(type='tforce_additions.preprocessors.Cast', dtype=tf.float32)]


# DQN agent
from tensorforce.agents import DQNAgent

# Architecture: see tensorforce/core/networks/layer.py for documentation
# and tensorforce/examples/configs for example setups
dqn_spec = [
        # Layer 1: fully connected, num states->num actions, LReLU activation
        dict(type='linear', size=num_pixels*num_actions_per_pixel),
        dict(type='nonlinearity', name='lrelu', alpha=0.2), # alpha: slope at x < 0
        
        # Layer 2: fully connected, num actions->num actions, LReLU activation
        dict(type='linear', size=num_pixels*num_actions_per_pixel),
        dict(type='nonlinearity', name='lrelu', alpha=0.2), 
]

dqn_summary_spec = deepcopy(tensorboard_summary_spec_generic)
dqn_summary_spec['directory'] = 'results/dqn/'

import pdb; pdb.set_trace()
# TODO: investigate actions_exploration parameter. Passed to Model superclass and to Exploration
# object to initialize an exploration strategy, not sure what exploration types are valid/what params
# NOTE: Arguments don't appear to allow configuring loss function other than huber_loss;
#       assuming uses MSVE of Q-value as in Mnih et al. 2015
dqn_agent = DQNAgent(states, actions, dqn_spec, discount=discount,
                     optimizer=adam_optimizer,
                     states_preprocessing=preprocessing_config, # Convert states to float32
                     memory=None, # Experience replay buffer; see tensorforce doc for default vals
                     target_sync_frequency=10000, # How often to update the target network
                     target_update_weight=1.0, # Has something to do with updating target network? Default val
                     summarizer=dqn_summary_spec
)

import pdb; pdb.set_trace()
runner = Runner(dqn_agent, env_tforce)
def ep_finished(runnr):
        # Notify environment to use current runner policy as rollout policy for reward estimation
        runnr.env.gym.rollout_policy = (lambda s: runnr.agent.act(s))
        if runnr.episode % 10 == 0:
                print("Finished episode {ep} after {ts} timesteps".format(ep=runnr.episode + 1, ts=runnr.timestep + 1))
                print("Episode reward: {}".format(runnr.episode_rewards[-1]))
                print("Average of last 10 rewards: {}".format(np.mean(runnr.episode_rewards[-10:])))
        return True

# TODO: Investigate how long to set max_episode_timesteps
import pdb; pdb.set_trace()
runner.run(num_episodes = 100, max_episode_timesteps=100, episode_finished=ep_finished)




# def deep_q_network(state, num_pixels, num_actions_per_pixel):
# 	with tf.variable_scope("deep_q_network") as scope:
#         # TODO: Convert this into an actual ConvNet
#         # Outputs 2 * num pixels, so mod by two to get whether it's black or white, and divide by 2 to get the num action.
# 		h0 = lrelu(dense(state, num_pixels * num_actions_per_pixel, name="dense1"))
# 		h1 = dense(h0, num_pixels * num_actions_per_pixel, name="dense2")
# 		h2 = tf.nn.softmax(h1, name="softmax1")
# 		return h2

# # Set up placeholder
# state_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, DIMENSION*DIMENSION], name='state')
# actual_reward = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1], name='actual_reward')
# action_selection = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='action_selection')

# # Set up loss function
# estimated_q_value = deep_q_network(state_placeholder, DIMENSION*DIMENSION, NUM_POSSIBLE_PIXEL_COLORS)
# objective_fn = tf.losses.mean_squared_error(actual_reward, tf.gather(estimated_q_value, action_selection, axis=1)[0])
# tf.summary.scalar("DQN Loss", objective_fn)
# # Set up optimizer
# q_optimizer = tf.train.AdamOptimizer(LEARNING_RATE, BETA1)
# grads = q_optimizer.compute_gradients(objective_fn)
# for grad, var in grads:
# 	if grad is not None:
# 		tf.summary.scalar(var.op.name + "/gradient", tf.reduce_mean(grad))
# train_q_network = q_optimizer.apply_gradients(grads)

# sess = tf.Session()

# merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter('tensorboard/',
# 									 sess.graph)

# # Initialize TF graph.
# sess.run(tf.global_variables_initializer())

# # Initialize epsilon greedy.
# epsilon_greedy = EPSILON_GREEDY_START

# # Training.
# for i in xrange(NUM_EPISODES):
# 	print("Episode num: " + str(i))
# 	curr_state = env.reset()
# 	episode_done = False

# 	# Perform one episode to finish.
# 	while not episode_done:
# 		# TODO: Use replay buffer to batch data.
# 		state_batch = np.array([curr_state])

# 		print("Current state: " + str(state_batch))

# 		# First do a forward prop to select the best action given the current state.
# 		q_value_estimates = sess.run([estimated_q_value], {state_placeholder: state_batch})

# 		print("Estimated Q Values: " + str(q_value_estimates))

# 		reward_batch = np.zeros((BATCH_SIZE,1))
# 		action_selection_batch = np.zeros((BATCH_SIZE,1))
# 		# For now, let's just do a loop over all the items in a batch...
# 		for b in xrange(BATCH_SIZE):
# 			s = state_batch[b]
# 			non_zero_states = np.argwhere(s != 0)
# 			print("Non zero states: " + str(non_zero_states))
# 			non_zero_actions = np.append(non_zero_states * 2, non_zero_states * 2 + 1)

# 			print("Q value estimates prior to validation: " + str (q_value_estimates[b][0]))
# 			q_value_estimates[b][0][non_zero_actions] = -100
# 			print("Q value estimates after validation: " + str (q_value_estimates[b][0]))
# 			max_idx = np.argmax(q_value_estimates[b][0])

# 			# TODO: Make this GLIE
# 			if np.random.rand() < epsilon_greedy:
# 				print("Selecting random action!")
# 				print(epsilon_greedy)
# 				random_choices = np.argwhere(s == 0)
# 				print("Selecting from zero-filled values")
# 				print(random_choices)
# 				rand_idx = np.random.randint(0, random_choices.size)
# 				# Smart random selection: only select pixel coords that are not yet selected.
# 				max_idx = random_choices[rand_idx][0] * 2
# 				max_idx += np.random.randint(0,2)
# 				print("Selecting: ")
# 				print(max_idx)

# 			# Remember max_idx is in the 1d flattened range of num_pixels * num_actions_per_pixel so
# 			# we have to do divison and modulus to fetch out the actual best pixel color and coordinate.
# 			best_pixel_color, best_pixel_coordinate = int((max_idx % 2) + 1), int(max_idx / 2)
# 			print("Best color: " + str(best_pixel_color))
# 			print("Best pixel coord: " + str(best_pixel_coordinate))

# 			# Using the best action, take it and get the reward and set the next state.
# 			next_state, reward, episode_done, _ = env.step_with_fill_policy([best_pixel_color, best_pixel_coordinate])
# 			print("Reward seen for this action: " + str(reward))
# 			print("Reward we thought this action would have: " + str(q_value_estimates[b][0][int(max_idx)]))
# 			reward_batch[b] = reward

# 			next_state_q_values = sess.run([estimated_q_value], {state_placeholder: np.array([next_state])})
# 			reward_batch[b] += np.max(next_state_q_values[0][0])
# 			print("Q(s',a') = " + str(np.max(next_state_q_values[0][0])))

# 			curr_state = next_state 
# 			action_selection_batch[b] = max_idx

# 		print("Training with the following:")
# 		print("\t\tState: " + str(state_batch))
# 		print("\t\tReward: " + str(reward_batch))
# 		print("\t\tReward we thought this action would have: " + str(q_value_estimates[b][0][int(max_idx)]))
# 		print("\t\tSelected action: " + str(action_selection_batch))
# 		# Given the reward, train our DQN.
# 		_, loss, summary = sess.run([train_q_network, objective_fn, merged], {state_placeholder: state_batch, actual_reward: reward_batch, action_selection: action_selection_batch})
# 		print("DQN Loss: " + str(loss))
# 		train_writer.add_summary(summary, i)
# 		print("")
# 	print("Episode finished. Rendering:")
# 	env.render()

# 	# Decay epsilon greedy value
# 	epsilon_greedy *= EPSILON_GREEDY_PER_EPISODE_DECAY

# # Try and draw an image with no randomness and no learning
# print("Drawing as a test")
# curr_state = env.reset()
# episode_done = False
# while not episode_done:
#         state_batch = np.array([curr_state])

#         print("Current state: " + str(state_batch))

# 	# First do a forward prop to select the best action given the current state.
# 	q_value_estimates = sess.run([estimated_q_value], {state_placeholder: state_batch})

# 	print("Estimated Q Values: " + str(q_value_estimates))

# 	reward_batch = np.zeros((BATCH_SIZE,1))
# 	action_selection_batch = np.zeros((BATCH_SIZE,1))
# 	# For now, let's just do a loop over all the items in a batch...
# 	for b in xrange(BATCH_SIZE):

# 		max_idx = np.argmax(q_value_estimates[b][0])
# 		best_action_num_value = q_value_estimates[b][0][max_idx]

# 		# Remember best_action_num_value is in the 1d flattened range of num_pixels * num_actions_per_pixel so
# 		# we have to do divison and modulus to fetch out the actual best pixel color and coordinate.
# 		best_pixel_color, best_pixel_coordinate = int((max_idx % 2) + 1), int(max_idx / 2)
# 		print("Best color: " + str(best_pixel_color))
# 		print("Best pixel coord: " + str(best_pixel_coordinate))

# 		# Using the best action, take it and get the reward and set the next state.
# 		curr_state, reward, episode_done, _ = env.step_with_fill_policy([best_pixel_color, best_pixel_coordinate])
# 		print("Reward seen for this action: " + str(reward))
# 		print("Reward we thought this action would have: " + str(q_value_estimates[b][0][int(max_idx)]))
# 		reward_batch[b] = reward
# 		action_selection_batch[b] = max_idx
# print("Test episode finished")
# env.render()


