import sys
import os 
import os.path as op
import json
import logging
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from collections import deque
from scipy import sparse
# import pandas as pd
import modin.pandas as pd
import numpy as np
import scipy 
from scipy import sparse
#from tqdm import tqdm
import geopy.distance as distance
from scipy.spatial import KDTree
import functools
#import joblib
import copy

sys.setrecursionlimit(10000)




def client_features(df, clt_spec_cols, group_key):
	""" build features for df using client-specific features

	Parameters
	----------
	df : pd.DataFrame or modin.pd.DataFrame

	clt_spec_cols : list
		column names of client specific features, like device name etc...
		details on these features in README

	group_key : str
		name of the column to group data

	Returns
	-------

	"""
	df_features = df[clt_spec_cols + [group_key]].groupby(group_key,
														  as_index=True,
														  sort=False)

	df_features = df_features.agg('first')

	return df_features

def bs_features(df, df_clt_features, bs_spec_cols, key1, key2):
	"""

	Parameters
	----------
	df
	df_clt_features : pd.DataFrame or moding.pd.DataFrame
		df containing client specific features, as output of `client_features`

	bs_spec_cols : list
	key1 : str
		first group key, should be 'messsageid'
	key2 : str
		second group key, should be 'bsid'
	Returns
	-------
	"""
	# first step: compute column names
	bsids = df[key2].unique()
	for feature in bs_spec_cols:
		for bsid in bsids:
			df_clt_features[feature + str(bsid)] = ([0] * df_clt_features.shape[0])

	# second step : group and fill the columns
	df_gp = df[[key1, key2] + bs_spec_cols].groupby([key1, key2], as_index=True)
	df_gp = df_gp.agg('first') # or mean?

	for gp_row in df_gp.iterrows():
		((msg_id, bsid), row) = gp_row
		for feature, value in row.iteritems():
			feature_name = feature + str(bsid)
			df_clt_features.loc[msg_id, feature_name] = value

	return df_clt_features

#@deprecated
dict_of_gby = {'rssi': ['bsid'],
				'freq': ['bsid'],
				'latitude_bs': ['bsid'],
				'longitude_bs': ['bsid'],
				'latitude': [''],
				'longitude': [''],
				'speed': [''],
				'dtid': [''], 
				'did': [''],}

def _build_features_dict(data_frame, features_of_interest):

	if 'latitude_bs' in list(features_of_interest.values())[0] or 'latitude_bs' in list(features_of_interest.values())[1]:
		sys.stdout.write(u"\u001b[4mDownloading base stations informations (take a coffee, can take a while)\u001b[0m \u001b[1m\u001b[0m \n")
		from sigfox.datamart_data import BasestationData
		basestationdata = BasestationData('/Users/kevinelgui/Thèse/Projet/Data/GeoDataFrame', **{'basestation_ids': bsid_unique}).dataframe
		basestationdata = basestationdata[basestationdata.objid.isin(bsid_unique)]
		data_frame['latitude_bs'] = data_frame.apply(lambda row: bs_dict[row['bsid']]['latitude'], 1)
		data_frame['longitude_bs'] = data_frame.apply(lambda row: bs_dict[row['bsid']]['longitude'], 1)
	#Verbose
	bsid_nunique = data_frame.bsid.nunique()
	bsid_unique = data_frame.bsid.unique()
	did_nunique = data_frame.did.nunique()
	sys.stdout.write(u"\u001b[4mFeatures of interest:\u001b[0m \u001b[1m {} \u001b[0m \n".format(
	features_of_interest.get('features_of_interest')))
	sys.stdout.write(u"\u001b[4mTarget:\u001b[0m \u001b[1m {} \u001b[0m \n".format(
	features_of_interest.get('target')))
	sys.stdout.write(u"\u001b[4mNb of base stations:\u001b[0m \u001b[1m {} \u001b[0m \n".format(bsid_nunique))
	sys.stdout.write(u"\u001b[4mNb of unique DeviceId train:\u001b[0m \u001b[1m {} \u001b[0m\n".format(str(did_nunique)))

	l0 = len(data_frame)


	st = {f.__add__(x) for f in features_of_interest['features_of_interest'] \
		  for x in dict_of_gby[f]}
	feature_name = set({})
	for f in features_of_interest['features_of_interest']:
		for x in dict_of_gby[f]:
			if x:
				g = data_frame.groupby(x).groups
				for gg in g:
					feature_name = feature_name.union({f.__add__(str(gg))})
			else:
				feature_name = feature_name.union({str(f)})
	
	target_name = set({})
	for f in features_of_interest['target']:
		for x in dict_of_gby[f]:
			if x:
				g = data_frame.groupby(x).groups
				for gg in g:
					target_name = target_name.union({f.__add__(str(gg))})
			else:
				target_name = target_name.union({str(f)})

	feature_dict = {name: i for i, name in enumerate(feature_name)}
	target_dict = {name: i for i, name in enumerate(target_name)}
	return (feature_dict, feature_name), (target_dict, target_name)

def my_parser(data, label, feature_dict, target_dict, features_of_interest):
	
	data_copy = pd.concat((data, label), 1)
	
	res = {'dict_X': feature_dict, 'dict_y': target_dict} 

	data_it = copy.deepcopy(data_copy)
	
	groupby_msgid = data_it.groupby(['messageid', 'time_ux'],
						sort=True, group_keys='time_ux')


	features_, I, J  = [], [], []
	
	y, I_output, J_output = [], [], []

	it = enumerate(groupby_msgid)  #.__iter__()

	for ind, ((_, time_ux), value) in it:
		time_ux = time_ux//1000
		did = value.did.unique()[0]

		"""
		fill y 
		"""
		for k in features_of_interest['target']:
			for gb_key in dict_of_gby[k]:
				if gb_key:
					for g, val_gb in value.groupby(gb_key):
							I_output.append(ind)
							J_output.append(target_dict[k.__add__(str(g))])
							y.append(val_gb[k].values[0])
				else:
					I_output.append(ind)
					J_output.append(target_dict[k])
					y.append(value[k].values[0]) 
		"""
		fill data 
		"""
		for k in features_of_interest['features_of_interest']:
			for gb_key in dict_of_gby[k]:
				if gb_key:
					for g, val_gb in value.groupby(gb_key):
							I.append(ind)
							J.append(feature_dict[k.__add__(str(g))])
							features_.append(1*(val_gb[k].max()< -1))
				else:
					I.append(ind)
					J.append(feature_dict[k])
					features_.append(1*(value[k].max()<-1))

	coo_matrix_ = sparse.coo_matrix((features_, (I, J)), shape=(len(groupby_msgid), len(feature_dict)), dtype=np.int64)
	coo_matrix_y = sparse.coo_matrix((np.array(y)[:,0], (I_output, J_output)), shape=(len(groupby_msgid), len(target_dict)), dtype=np.float32)
	res.update({'X': coo_matrix_, 'y': coo_matrix_y })

	return res


