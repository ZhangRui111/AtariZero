import os

from dmc import parser, train


if __name__ == '__main__':
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices  # Set CUDA_VISIBLE_DEVICES
    train(flags)
