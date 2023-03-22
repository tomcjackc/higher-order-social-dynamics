#%%
import numpy as np
def r2_score(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0, dtype=np.float64)
    return 1-(numerator/denominator)

print(r2_score(np.array([1,2,3,4,5]), np.array([1,2,3,4,5])))
#%%