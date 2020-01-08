import numpy as np
import pandas as pd
from ipdb import set_trace


ac = ['a1','a2','a3']
df = pd.DataFrame(columns = ['date']+ac)
print(df.shape)
print(','.join(df.columns.values))
set_trace()
#df['date'] = '2019'
for i in range(np.shape(a)[1]):
    df[name[i]] = a[:,i]
df['ax'] = '2019-01-02'
df = df.fillna(0)
print(df)
set_trace()

a = np.array([1,2,3,4,5])
print(np.shape(a))
set_trace()


t = np.matrix([1,2,3]).T
print(np.shape(t))
t = np.zeros(3)
print(np.shape(t))
print(np.hstack(([0],t)))

df = pd.DataFrame(columns = ['a','b','c','d','f','e','k'])
columns = ['a','c','b']
print(columns.sort())
dfn = df[columns]
print(dfn)
