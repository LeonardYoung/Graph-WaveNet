import os
import sys
import subprocess


def run_LSTM(env):
    if env == 'server':
        subprocess.run(["/home/s304/miniconda3/envs/ysj_torch/bin/python",'-u',
                    "/media/s304/Data/yangsj/project/waveNet2/water/waterLSTM.py"])
    else:
        os.system('python water/waterLSTM.py')


def run_waveNet(env):
    pass


if __name__ == '__main__':
    run_LSTM(sys.argv[1])

