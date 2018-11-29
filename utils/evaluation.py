import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from geopy.distance import vincenty
import pandas as pd
#import seaborn as sns



def vincenty_df(y_test, y_pred): #, columns=['latitude', 'longitude']):
	""" returns pointwiwe Vincenty distance when input is in a DataFrame

	Parameters
	----------
	y_test : pd.DataFrame, shape(n, 2)
		test DF, contains true coordinates

	y_pred : pd.DataFrame, shape(n, 2)
		predicted DF, contains predicted coordinates


	Returns
	-------
	m : ndarray
		each entry is Vincenty distance between the predicted and the true
		point
	"""
	assert(y_pred.shape == y_test.shape)

	vin_vec_dist = []
	for i in range(y_pred.shape[0]):
		pred = (y_pred.iloc[i, 0], y_pred.iloc[i, 1])
		test = (y_test.iloc[i, 0], y_test.iloc[i, 1])

		vin_vec_dist.append(vincenty(test, pred).km)

	return np.array(vin_vec_dist)


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


def criterion(y_true, y_pred, is_df):
	"""

	Parameters
	----------
	y_pred
	y_true
	is_df : bool
		True if input is on pd.DataFrame format with 2 columns
		else is as array of tple

	Returns
	-------

	"""
	if is_df:
		error_vector = vincenty_df(y_true, y_pred)
	else:
		error_vector = vincenty_vec(y_true, y_pred)
	return np.percentile(error_vector, 90)


def plot_error(y_pred, y_true, is_df):
	if is_df:
		error_vector = vincenty_df(y_true, y_pred)
	else:
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







