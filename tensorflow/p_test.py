import matplotlib.pyplot as plt
from matplotlib.dates import (YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange)
import numpy as np
import datetime

# Fixing random state for reproducibility
np.random.seed(19680801)


# tick every 5th easter
rule = rrulewrapper(YEARLY, byeaster=1, interval=5)
loc = RRuleLocator(rule)
formatter = DateFormatter('%m-%d-%y')

date1 = datetime.datetime.strptime("1952-01-01", '%Y-%m-%d').date()
date2 = datetime.datetime.strptime("2004-04-12", '%Y-%m-%d').date()
delta = datetime.timedelta(days=5)

dates = ["1992-01-01", "1992-01-02", "1992-01-05", "1992-01-07", "1992-01-09"]
#dates = drange(date1, date2, delta)
s = np.random.rand(len(dates))  # make up some random y values


fig, ax = plt.subplots()
plt.plot_date(dates, s)
#ax.xaxis.set_major_locator(loc)
#ax.xaxis.set_major_formatter(formatter)
#ax.xaxis.set_tick_params(rotation=30, labelsize=10)

plt.show()