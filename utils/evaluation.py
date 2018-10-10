import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from geopy.distance import vincenty
import pandas as pd
#import seaborn as sns



def vincenty_vec(coord1, coord2):
	"""
	Returns the pointwise Vincenty distance between two GPS-coords arrays

	Parameters
	----------
	coord1 : np.array ['latitude', 'longitude']
	coord2 : np.array ['latitude', 'longitude']

	Returns
	-------
	m : ndarray 
	    m[i] = vincenty_distance(coord1[i], coord2[i])

	"""
	assert(coord1.shape == coord2.shape)
	vin_vec_dist = [(vincenty(z1, z2)).km for (z1, z2) in zip(coord1, coord2)]
	return vin_vec_dist


def criterion(y_pred, y_true):
	error_vector = vincenty_vec(y_pred, y_true)
	return np.percentile(error_vector, 90)


def plot_error(y_pred, y_true):
	error_vector = vincenty_vec(y_pred, y_true)
	
	f = plt.figure()
	ax = f.add_subplot(111)
	plt.hist(error_vector, cumulative=True, histtype='step', density=True, bins=500)

	plt.vlines(x=np.percentile(error_vector, 90), ymin=0, ymax=1, colors='r', label='criterion')
	plt.xlabel('Distance Error (km)')
	plt.ylabel('Cum Proba (%)')
	plt.xlim(-.1, 100)
	f.legend()
	f.show()







