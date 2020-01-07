
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from ipdb import set_trace

#print(pd.Timestamp(datetime.today()).strftime('%Y-%m-%d'))

t = pd.DataFrame()
t['a'] = [1,2,6,2,3,6,1,4]
t['b'] = [3,4,5,2,3,4,4,5]
t = t.groupby('a')
t = t.get_group(1)
print(t)
set_trace()
print(t.std()['a'])
#t = np.matrix([[1,2,3],[1,2,3],[2,3,4],[3,6,7]])
t = np.matrix(t)
print(np.shape(t))
print(np.cov(t.T))
