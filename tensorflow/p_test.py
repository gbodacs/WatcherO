import matplotlib.pyplot as plt
from matplotlib.dates import (YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange)
import numpy as np
import datetime
import glob

for file in glob.glob("*.csv"):
    print(file)