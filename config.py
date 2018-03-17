import drawing_env.envs.config as de_cfg

# General parameters
FULL_DIMENSION = de_cfg.MNIST_DIMENSION
LOCAL_DIMENSION = 7
NUM_POSSIBLE_PIXEL_COLORS = de_cfg.NUM_POSSIBLE_PIXEL_VALUES-1
EPISODE_LENGTH = FULL_DIMENSION*FULL_DIMENSION*2
NUM_STATES = LOCAL_DIMENSION*LOCAL_DIMENSION
NUM_EPISODES = 100000

# Parameters for DQN
BETA1 = 0.5
LEARNING_RATE = 5e-3
BATCH_SIZE = 1

# Parameters for PG
PG_SUMMARY_FREQ = 1
PG_DRAW_FREQ = 1
PG_LR = 5e-3
PG_EPISODE_LENGTH = FULL_DIMENSION * FULL_DIMENSION
PG_NDRAWS_PER_BATCH = 10
PG_BATCH_NSTEPS = PG_NDRAWS_PER_BATCH*PG_EPISODE_LENGTH
PG_NUM_BATCHES = 500
PG_GSTEPS_PER_DSTEPS = 5
