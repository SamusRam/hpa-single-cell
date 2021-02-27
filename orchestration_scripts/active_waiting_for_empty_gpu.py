import GPUtil
import time

# in case some process requiring GPU is just getting started
time.sleep(180)
while len(GPUtil.getAvailable(order='PCI_BUS_ID', maxMemory=0.05, includeNan=False, excludeID=[0], excludeUUID=[])) == 0:
    time.sleep(60)
