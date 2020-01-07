import numpy as np
import pandas as pd

t = np.matrix([1,2,3]).T
print(np.shape(t))
t = np.zeros(3)
print(t)

df = pd.DataFrame(columns = ['a','b','c','d','f','e','k'])
columns = ['a','c','b']
print(columns.sort())
dfn = df[columns]
print(dfn)
