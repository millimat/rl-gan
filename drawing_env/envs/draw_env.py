import numpy as np
import tensorflow as tf

from gym import Env, spaces
from gym.utils import seeding
from discriminator import RLDiscriminator

# Zero means the pixel color hasn't been selected yet.
# One means the pixel color is white.
# Two means the pixel color is black.
NUM_POSSIBLE_PIXEL_VALUES = 3

UNFILLED_REWARD_FACTOR = 3

GAMMA = 0.999
REWARD_CONSTANT = 50

class DrawEnv(Env):
	def __init__(self, dimension=3):
		# Dimensions of the drawing. Note that the drawing will always be
		# a square, so the dimension is both the height and the width.
		self.dimension = dimension

		# MultiBinary gives us a zero or one action for each of (len)
		# passed in as an argument. So we pass in the total number of pixels,
		# dimension squared, to get the total action space.
		#
		# Note that the actions are to select 0 (meaning color white) or 1
		# (meaning color black) for a given pixel.
                
                # TODO (matt): switch MultiBinary to MultiDiscrete([NUM_POSSIBLE_PIXEL_VALUES-1, dimension**2]).
                # TODO (matt): switch unknown pixel from 0 to NUM_POSSIBLE_PIXEL_VALUES-1 so that actions
                # correspond to coloring pixels specific colors
		self.action_space = spaces.MultiBinary(dimension*dimension)

		# The first value in the action tuple is the pixel.
		#
		# The second value in the action is the pixel coordinate, flattened
		# into a 1d value.
                
                # TODO (matt): Should the state be the current state of all pixels?
		self.observation_space = spaces.MultiDiscrete([NUM_POSSIBLE_PIXEL_VALUES, dimension*dimension])

		self._reset_pixel_values()

		self.sess = tf.Session()
		self.rl_discriminator = RLDiscriminator(self.sess, dimension, dimension, 1)
		self.discount_factor = 1

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

	def step_with_fill_policy(self, a, fill_policy=None):
		# First check if action is valid.
		assert a[0] < NUM_POSSIBLE_PIXEL_VALUES, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-1) + "]. Current invalid action: " + str(a)
		assert a[0] >= 0, "Pixel value for an action must fall in range: [0," + str(NUM_POSSIBLE_PIXEL_VALUES-1) + "]. Current invalid action: " + str(a)
		assert a[1] < self.dimension * self.dimension, "Pixel coordinate for an action must fall in range [0," + str(self.dimension*self.dimension-1) + "]. Current invalid action: " + str(a)
		assert a[1] >= 0, "Pixel coordinate for an action must fall in range [0," + str(self.dimension*self.dimension-1) + "]. Current invalid action: " + str(a)

		# If the action is None or if it sets a pixel to 0 or if this coordinate is already set.
		if a is None or a[0] == 0 or self.pixel_values[a[1]] != 0:
			if self.pixel_values[a[1]] != 0:
				print("Trying to set pixel coordinate " + str(a[1]) + " to " + str(a[0]) + " when a value is already selected for it: " + str(self.pixel_values[a[1]]))
			return self.pixel_values, -10, False, {}

		# Set the pixel value in our state based on the action.
		self.pixel_values[a[1]] = a[0]

		# We can do this because np.all returns true unless there is any zero-valued pixel (meaning
		# a pixel color hasn't been selected for one coordinate).
		done = np.all(self.pixel_values)

		fake_image, num_unfilled_pixels = self._fill_remaining_pixels(fill_policy)
		print("Had to fill in " + str(num_unfilled_pixels) + " pixels.")

		# Update discount factor
		self.discount_factor *= GAMMA

		if done:
			fake_prob = self.rl_discriminator.train(fake_image, True)
		else:
			fake_prob = self.rl_discriminator.get_disc_loss(fake_image, True)

		return self.pixel_values, self._compute_reward(fake_prob, num_unfilled_pixels), done, {}

	def _reset_pixel_values(self):
		# The actual pixel values of the drawing. We start out with all values
		# equal to zero, meaning none of the pixel colors have been selected yet.
		self.pixel_values = np.full(self.dimension*self.dimension, 0)

	# The reward is a function of how much we were able to trick the discriminator (i.e. how
	# high the fake_prob is) and how many pixels had to be filled in.
	def _compute_reward(self, fake_prob, num_unfilled_pixels):
		# For now, we try the following reward:
		# - We want the fake_prob to be correlated with the reward
		# - We want the num_unfilled_pixels to be inversely weighted with the reward
		# So we just try fake_prob / (num_unfilled_pixels + 1).
		# Note the +1 is needed so we don't divide by zero.
		# TODO: Experiment with this.
		return (fake_prob - 1) / ((num_unfilled_pixels + 1) * UNFILLED_REWARD_FACTOR) * self.discount_factor * REWARD_CONSTANT

	# Returns a copied state with the remaining pixels filled in according to the current policy,
	# and also returns the number of pixels that had to be filled in.
	def _fill_remaining_pixels(self, fill_policy=None):
		# TODO, actually do this properly.
		copied_pixels = np.copy(self.pixel_values)
		num_unfilled_pixels = 0
		for i in xrange(copied_pixels.size):
			# For now, if we find an empty pixel, just fill it in with a random pixel.
			if copied_pixels[i] == 0:
				copied_pixels[i] = np.random.randint(1,3)
				num_unfilled_pixels += 1
		return copied_pixels, num_unfilled_pixels


