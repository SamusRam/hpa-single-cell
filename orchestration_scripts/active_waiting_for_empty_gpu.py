import GPUtil
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu-i', type=int, default=0)
parser.add_argument('--sleep-sec', type=int, default=60)

args = parser.parse_args()
gpu_i = args.gpu_i
sleep_duration = args.sleep_sec


# in case some process requiring GPU is just getting started
time.sleep(180)
while gpu_i not in GPUtil.getAvailable(order='PCI_BUS_ID', maxMemory=0.05, includeNan=False, excludeID=[], excludeUUID=[]):
    time.sleep(sleep_duration)
