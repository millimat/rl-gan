import numpy as np
import tensorflow as tf

from gym import Env, spaces
from gym.utils import seeding
from discriminator import RLDiscriminator

# Zero means the pixel color hasn't been selected yet.
# One means the pixel color is white.
# Two means the pixel color is black.
NUM_POSSIBLE_PIXEL_VALUES = 3
UNFILLED = 0

UNFILLED_REWARD_FACTOR = 3

GAMMA = 0.999
REWARD_CONSTANT = 50

class DrawEnv(Env):
	def __init__(self, dimension=3):
		# Dimensions of the drawing. Note that the drawing will always be
		# a square, so the dimension is both the height and the width.
		self.dimension = dimension

                # Represent one action per pixel per possible coloring of that pixel.
                self.n_actions = dimension*dimension*(NUM_POSSIBLE_PIXEL_VALUES-1)
                self.action_space = spaces.Discrete(self.n_actions)                
                
                # The state is the current filling of the image: one value for each pixel.
		self.observation_space = spaces.MultiDiscrete([NUM_POSSIBLE_PIXEL_VALUES] * (dimension*dimension))

		self._reset_pixel_values()

                # Setup discriminator
		self.sess = tf.Session()
		self.rl_discriminator = RLDiscriminator(self.sess, dimension, dimension, 1)

                # Determine number of rollouts for estimation of rewrd
                self.rollout_policy = (lambda s: self.action_space.sample())
                self.num_rollouts = 10 

              
                
	def render(self, mode='human'):
		# TODO: Write this out to an actual file, and convert the pixel values to a format
		# that will allow the file to render as an actual image.

		# Right now, we're simply printing out the pixel_values to stdout.
		print("--------------------------")
		print("\n")
		for row in xrange(self.dimension):
			row_str = ''
			for col in xrange(self.dimension):
				row_str += str(self.pixel_values[(row * self.dimension) + col]) + " "
			print(row_str + "\n")
		print("--------------------------")

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random()
		return [seed]

        
	# Note this does not reset the discriminator parameters.
	def reset(self):
		self._reset_pixel_values()
		self.discount_factor = 1
		return self.pixel_values

        
        # Convert z in {0, ..., dimension**2 - 1} to a pixel position (i,j).
        def _pixel_1d_to_2d(z):
                return (z/dimension, z%dimension)

        
        # Convert an action index a in {0, ..., dimension**2 * (NUM_POSSIBLE_PIXEL_VALUES-1) - 1}
        # to a 1d image position and pixel value.
        # Since color 0 is reserved for UNFILLED, the first fill color is 1.
        def _action_to_1d_pixel_color(self, a):
                assert 0 <= a < self.n_actions,\
                        'action {} outside of valid range [0,{}]'.format(a, self.n_actions)
                return (a/NUM_POSSIBLE_PIXEL_VALUES, 1 + a%NUM_POSSIBLE_PIXEL_VALUES)


        # Attempt to take action a by filling a pixel.
        # If the pixel has already been filled, do not overwrite it.
        # Return the pixel filled and what we filled it with.
        def _fill_one_pixel(self, a, target=None):
                if target is None:
                        target = self.pixel_values
                z, fill = self._action_to_1d_pixel_color(a)
                if target[z] == UNFILLED:
                        target[z] = fill
                return z, fill
                
                
        
        # Expects a in {0, ..., dimension**2 * NUM_POSSIBLE_PIXEL_VALUES - 1}.
        def step(self, a):
                self._fill_one_pixel(a)
                done = np.all(self.pixel_values != UNFILLED)

                rollout_scores = []
                num_unfilled_pixels = -1
                for _ in xrange(self.num_rollouts):
                        rollout_image, num_unfilled_pixels = self._rollout()
                        rollout_score = self.rl_discriminator.train(rollout_image) if done\
                                        else self.rl_discriminator.get_disc_loss(rollout_image)
                        rollout_scores.append(rollout_score)

                return self.pixel_values, self._compute_reward2(np.mean(rollout_scores), num_unfilled_pixels),\
                        done, {}
               
        
	def _reset_pixel_values(self):
		# The actual pixel values of the drawing. We start out with all values
		# equal to zero, meaning none of the pixel colors have been selected yet.
		self.pixel_values = np.full(self.dimension*self.dimension, UNFILLED)
                

	# # The reward is a function of how much we were able to trick the discriminator (i.e. how
	# # high the fake_prob is) and how many pixels had to be filled in.
	# def _compute_reward(self, fake_prob, num_unfilled_pixels):
	# 	# For now, we try the following reward:
	# 	# - We want the fake_prob to be correlated with the reward
	# 	# - We want the num_unfilled_pixels to be inversely weighted with the reward
	# 	# So we just try fake_prob / (num_unfilled_pixels + 1).
	# 	# Note the +1 is needed so we don't divide by zero.
	# 	# TODO: Experiment with this.
	# 	return (fake_prob - 1) / ((num_unfilled_pixels + 1) * UNFILLED_REWARD_FACTOR) * self.discount_factor * REWARD_CONSTANT

        
        # Reward the agent with the mean discounted discriminator score from several rollouts.
        # TODO (matt): consider discounting by number of actions in rollout instead to penalize slowness,
        #              and discount each score with individual num steps before computing mean
        def _compute_reward2(self, rollout_score, num_unfilled):
                return GAMMA**num_unfilled * rollout_score
        
        
        # Simulate filling in the rest of the image with a given policy.
        # fill_policy should be a function mapping states to valid actions.
        # If fill_policy does not systematically complete remaining unfilled pixels,
        # the policy may not generate a complete image in a finite number of steps,
        # so we allow the rollout to proceed for a limited number of steps before giving up.
        def _rollout(self, fill_policy=None, max_rollout_duration=None):
                if fill_policy is None:
                        fill_policy = self.rollout_policy
                if max_rollout_duration is None:
                        max_rollout_duration = 2*self.dimension**2
                        
                pixels_copy = np.copy(self.pixel_values)
                remaining = set(np.where(self.pixel_values == UNFILLED)[0])
                n_unfilled = len(remaining)
                for _ in xrange(max_rollout_duration):
                        action = fill_policy(pixels_copy)
                        z, fill = self._fill_one_pixel(action, target=pixels_copy)
                        if z in remaining:
                                remaining.remove(z)
                        if not remaining:
                                break

                return pixels_copy, n_unfilled 
