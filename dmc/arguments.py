import argparse

parser = argparse.ArgumentParser(description='AtariZero: PyTorch AI of Atari Game')

# General Settings
parser.add_argument('--env', default='Pong-ram-v0', type=str,
                    choices=['Pong-v0', 'Pong-ram-v0'],
                    help='Name of the atari game')
parser.add_argument('--save_interval', default=120, type=int,
                    help='Time interval (in minutes) at which to save the model')

# Training settings
parser.add_argument('--gpu_devices', default='0', type=str,
                    help='Which GPUs to be used for training')
parser.add_argument('--num_actor_devices', default=1, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=5, type=int,
                    help='The number of actors for each simulation device')
parser.add_argument('--num_threads', default=4, type=int,
                    help='Number learner threads')
parser.add_argument('--training_device', default=0, type=int,
                    help='The index of the GPU used for training models')
parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--save_dir', default='logs/checkpoints',
                    help='Root dir where experiment data will be saved')

# Hyper-parameters
parser.add_argument('--total_frames', default=100000000000, type=int,
                    help='Total environment frames to train for')
parser.add_argument('--exp_epsilon', default=0.01, type=float,
                    help='The probability for exploration')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Learner batch size')
parser.add_argument('--unroll_length', default=100, type=int,
                    help='The unroll length (time dimension)')
parser.add_argument('--num_buffers', default=50, type=int,
                    help='Number of shared-memory buffers')

# Optimizer settings
parser.add_argument('--learning_rate', default=0.00002, type=float,
                    help='Learning rate')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum')
parser.add_argument('--epsilon', default=1e-5, type=float,
                    help='RMSProp epsilon')
