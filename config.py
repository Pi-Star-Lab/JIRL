# Raw camera input
CAMERA_HEIGHT = 112
CAMERA_WIDTH = 112
CAMERA_RESOLUTION = (CAMERA_WIDTH, CAMERA_HEIGHT)
MARGIN_TOP = 32 # CAMERA_HEIGHT // 3
# Region Of Interest
# r = [margin_left, margin_top, width, height]
ROI = [0, MARGIN_TOP, CAMERA_WIDTH, CAMERA_HEIGHT - MARGIN_TOP]

# Input dimension for VAE
IMAGE_WIDTH = ROI[2]
IMAGE_HEIGHT = ROI[3]
N_CHANNELS = 3
INPUT_DIM = (IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)

# Reward parameters
THROTTLE_REWARD_WEIGHT = 0.1
JERK_REWARD_WEIGHT = 0.0

# very smooth control: 10% -> 0.2 diff in steering allowed (requires more training)
# smooth control: 15% -> 0.3 diff in steering allowed
MAX_STEERING_DIFF = 0.15
# Negative reward for getting off the road
REWARD_CRASH = -10
# Penalize the agent even more when being fast
CRASH_SPEED_WEIGHT = 5

# Symmetric command
MAX_STEERING = 1
MIN_STEERING = - MAX_STEERING

# Simulation config
MIN_THROTTLE = 0.45
# max_throttle: 0.6 for level 0 and 0.5 for level 1
MAX_THROTTLE =  0.6
# Number of past commands to concatenate with the input
N_COMMAND_HISTORY = 0
# Max cross track error (used in normal mode to reset the car)
MAX_CTE_ERROR = 2.0

# Action repeat
FRAME_SKIP = 1
Z_SIZE = 512  # Only used for random features
TEST_FRAME_SKIP = 1

BASE_ENV = "JetVae-v0"
ENV_ID = "JetVae-v0"
# Params that are logged
SIM_PARAMS = ['MIN_THROTTLE', 'MAX_THROTTLE', 'FRAME_SKIP',
              'MAX_CTE_ERROR', 'N_COMMAND_HISTORY', 'MAX_STEERING_DIFF']

# Jet Racer
STEERING_GAIN = 1
STEERING_BIAS = 0
