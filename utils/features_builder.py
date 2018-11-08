import sys
import os 
import pandas as pd
# import modin.pandas as pd
import numpy as np
import scipy 
from scipy import sparse
from multiprocessing import Pool, Value
#from tqdm import tqdm
import copy

sys.setrecursionlimit(10000)


class Builder(object):
	""" class to build feature matrix

	msgid : str
		column name for message id

	bsid : str
		column name for  basestation id

	clt_cols : str
		name of the columns of client specific features

	bs_cols : str
		name of the columns of basestation specific features
	"""
	def __init__(self, msgid, bsid, clt_cols, bs_cols, verbose):
		self.msgid = msgid
		self.bsid = bsid
		self.clt_cols = clt_cols
		self.bs_cols = bs_cols
		# to keep track of work
		self.counter = 1 #Value('i', 1)
		self.verbose = verbose
		# number of samples to classify
		self.unique_messages = None
		# df containing feature matrix after engineering
		self.df_features_ = None
		# df groupby message ids
		self.df_gp_ = None
		# bs feature names
		self.bs_feature_names = None


	def client_features(self, df):
		""" build features for df using client-specific features

		Parameters
        ----------
        df : pd.DataFrame or modin.pd.DataFrame

        Returns
        -------
        @TODO check return
        df
        same as input

        """
		df_features = \
			df[self.clt_cols + [self.msgid]].groupby(self.msgid,
													 as_index=True,
													 sort=False)

		df_features = df_features.agg('first')

		self.unique_messages = df[self.msgid].nunique()

		self.df_features_ = df_features

		return df

	def gb_bs_features(self, df):#, verbose=10):
		"""

		Parameters
		----------
		df

		verbose

		Returns
		-------
		"""
		# first step: compute column names
		self.bs_feature_names = []
		bsids = df[self.bsid].unique()
		for feature in self.bs_cols:
			for bsid in bsids:
				# self.df_features_[feature + str(bsid)] = (
				# 			[0] * self.df_features_.shape[0])
				self.bs_feature_names.append(feature + str(bsid))

		# second step : group and fill the columns
		df_gp = df[[self.msgid, self.bsid] + self.bs_cols]\
			.groupby([self.msgid, self.bsid], as_index=True)
		df_gp = df_gp.agg('first')  # or mean?

		n_gp = df_gp.shape[0]
		print('Shape of df_groupby: {}'.format(df_gp.shape))

		self.df_gp_ = df_gp

		return df_gp


	def fast_bs_features(self, n_jobs=2):
		""" script to fetch bs features

		Parameters
		----------
		verbose

		Returns
		-------

		"""
		if n_jobs == -1:
			n_jobs = None
			chunksize = int(self.unique_messages / 200)
		else:
			chunksize = int(self.df_gp_.shape[0] / n_jobs)

		global res
		with Pool(n_jobs) as pool:
			res = pool.map(self._local_bs_feature, self.df_gp_.iterrows(),
						   chunksize=chunksize)

		df_res = pd.DataFrame(res, columns=[self.msgid] + self.bs_feature_names)
		df_res = df_res.groupby(self.msgid, as_index=True).mean().fillna(0).to_sparse()
		#.fillna(0).to_sparse()
		# return self.df_features_

		return df_res


	def _local_bs_feature(self, gb_row):
		"""

		Parameters
		----------
		gb_row: tuple, ((msgid, bsid), row)

		Returns
		-------
		"""
		((msgid, bsid), row) = gb_row
		self.counter += 1
		if self.counter % self.verbose == 0:
			print('{:.3f}% of database parsed'.format((self.counter /self.unique_messages)*100 ))
		res = {self.msgid: msgid}
		for feature, value in row.iteritems():
			feature_name = feature + str(bsid)
			# self.df_features_.loc[msgid, feature_name] = value
			res[feature_name] = value

		return res
		# return pd.DataFrame(res, index=res[msgid], columns=self.bs_feature_names)



	# @deprecated
	def bs_features(self, df, verbose= 10):
		""" brute force construction of features, complexity O(npS)
		n number of messages, p = bs-spec features, S number of bs

		Parameters
		----------
		df
		verbose
		Returns
		-------
		"""
		count = 1
		for gp_row in self.df_gp_.iterrows():
			((msg_id, bsid), row) = gp_row
			count += 1
			if count % verbose == 0:
				print('Building feature for msg {}'.format(msg_id))
				print('{:.3f}% of df_gp done'.format((count / n_gp) * 100))
			for feature, value in row.iteritems():
				feature_name = feature + str(bsid)
				self.df_features_.loc[msg_id, feature_name] = value

		return self.df_features_

		








# @deprecated
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


