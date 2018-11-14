import joblib
import os
import sys
import numpy as np

clf = joblib.load('/home/namvh/Documents/do an may hoc/file/pridict/modelSVM.joblib')
test = np.load('/home/namvh/Documents/do an may hoc/file/pridict/feature/img.1.1.npy')
result = clf.predict(np.reshape(test, (1, -1)))
print(result)
