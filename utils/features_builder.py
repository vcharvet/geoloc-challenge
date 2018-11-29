"""
Our input datasets contain several rows per message, each message is present the
number of times it has been received by an antenna.
statics_params are the same for all couple (message, bs) bu param1, ... paramp
are different for every couple:
input [mssgid, bsid, static_params,  param1, ... paramp]  : that is p+p_0 parameters

This class intends to transform the matrix into
output[msgid, static_params, param1_bsid1, ..., paramp_bsid1, ...  paramp_bsid_I
where there are I base stations
it transform into a matrix with p * I + p_0 features
"""

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

	clt_cols : list of str
		name of the columns of client specific features
		Correspond to 'static' parameters

	bs_cols : list of str
		name of the columns of basestation specific features
		corredpond to param1, ... paramp

	verbose : bool
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
		""" build features for df using client-specific features, ie
		static features indicated in self.clt_cols

		Parameters
        ----------
        df : pd.DataFrame or modin.pd.DataFrame
        	raw input dataframe

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

	def gb_bs_features(self, df):
		""" it fetches the name of the parameters as concatenation of strings
		'param_name' + '_bsid' for simplicity

		Parameters
		----------
		df : pd.DataFrame
			raw input dataframe

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
		n_jobs : int, default=2
			number of CPUs to use for parallel computation. If -1, all available
			cores are used
		Returns
		-------
		pd.DataFrame
			the feature matrix with messageid as index and with concatenated
			featurei_bsid feature columns

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
		"""	fetches a new row of bsid features
		Function is private and only intended to use to ease parallel computation

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
