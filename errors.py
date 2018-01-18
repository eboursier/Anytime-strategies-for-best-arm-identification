import numpy as np

def minscore(MAB, reco):
	"""
	for reco the vector of recommandations of length(T, m), return the minimal mean at
	each timestamp t of the arms in J (ie the m recommended arms)
	"""
	means = np.zeros((len(reco), len(reco[0])))
	for i, rec_t in enumerate(reco):
		for j, arm in enumerate(rec_t):
			means[i, j] = MAB[arm].mean
	return np.min(means, axis=1)

def sumscore(MAB, reco):
	"""
	for reco the vector of recommandations of length(T, m), return the sum of the means at
	each timestamp t of the arms in J (ie the m recommended arms)
	"""
	means = np.zeros((len(reco), len(reco[0])))
	for i, rec_t in enumerate(reco):
		for j, arm in enumerate(rec_t):
			means[i, j] = MAB[arm].mean
	return np.sum(means, axis=1)