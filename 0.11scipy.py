#scipy mean percentile 
import numpy as np
from scipy import stats

data = np.random.normal(loc=0, scale=1, size=1000)
mean = stats.tmean(data)
percentile_50 = np.percentile(data, 50)
print("Mean:", mean)
print("50th Percentile:", percentile_50)    