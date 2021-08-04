import os

from douzero.dmc import parser

if __name__ == '__main__':
    flags = parser.parse_args()

    #  Set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices

