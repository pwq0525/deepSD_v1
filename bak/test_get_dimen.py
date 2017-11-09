from read_nc import *
import numpy as np

name = 'label_141'
d = np.array(get_dimen(name))
for i in range(len(d)):
	print(d[i][:10])
