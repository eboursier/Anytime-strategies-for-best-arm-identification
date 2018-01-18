import numpy as np

class GaussianPi():

	"""
	This class represents the distribution Pi_n.
	The distribution of Pi_n is computed iteratively using this class
	and a sample can be easily drawn from this class.
	"""

	def __init__(self, k, sigma=1/2):
		"""
		k is the number of arms	
		"""	
		# we will choose pi_1 as a uniform distribution between 0 and 1
		# because pi_1 is constant here, pi_n corresponds at a gaussian distribution at each time
		# for each theta_i. Thus, we will maintain mu_i, the expectations of theta_i drawn by \Pi_n
		# after computation for this model, we find that :
		# mu_i will be equal to the empirical average of the rewards when drawing the arm i
		self.mu = np.zeros(k)
		self.sigma = sigma
		self.sigmas = np.inf*np.ones(k)
		self.draws = np.zeros(k)

		# indicate for which arms the number of draws is 0 (ie sigma is inf)
		self.isinf = np.arange(k)


	def update(self, r, d):
		"""
		Update the distribution of GaussianPi with the new reward r from the draw d
		"""

		self.mu[d] = self.mu[d]*self.draws[d]/(self.draws[d]+1) + r/(self.draws[d]+1)
		self.sigmas[d] = self.sigma/np.sqrt(self.draws[d]+1)
		if d in self.isinf:
			self.isinf = np.setdiff1d(self.isinf, d)
		self.draws[d] += 1

	def sample(self):
		"""
		return a sample theta following the law Pi_n
		"""
		s = np.random.normal(self.mu, self.sigmas)
		# we now need to draw uniform values for the indices of infinite sigmas
		u = np.random.rand(len(self.isinf))
		s[self.isinf] = u

		return s

def TTTS(MAB, T=10000, beta=0.5, PosteriorDistrib=GaussianPi):
	k = len(MAB)
	Pi_n = PosteriorDistrib(k)
	reco = []
	for t in range(T):
		theta=Pi_n.sample()
		I = np.argmax(theta)
		B = np.random.binomial(1, beta)
		if B==1:
			# play I
			r = MAB[I].sample()
			Pi_n.update(r, I)

		else:
			J = I
			while (J==I):
				theta=Pi_n.sample()
				J = np.argmax(theta)
			# play J
			r = MAB[J].sample()
			Pi_n.update(r, J)
		# we recommend the best arm in posterior mean
		posterior_best = np.argmax(Pi_n.mu)
		reco.append([posterior_best])

	return reco

def TTTS2(MAB, T=10000, beta=0.5, PosteriorDistrib=GaussianPi):
	k = len(MAB)
	Pi_n = PosteriorDistrib(k)
	reco = []
	for t in range(T):
		theta=Pi_n.sample()
		I = np.argmax(theta)
		B = np.random.binomial(1, beta)
		if B==1:
			# play I
			r = MAB[I].sample()
			Pi_n.update(r, I)

		else:
			theta[I]=-np.inf
			J = np.argmax(theta)
			# play J
			r = MAB[J].sample()
			Pi_n.update(r, J)
		# we recommend the best arm in posterior mean
		posterior_best = np.argmax(Pi_n.mu)
		reco.append([posterior_best])

	return reco

def TTPS(MAB, T=10000, M=20
	, beta=0.5, PosteriorDistrib=GaussianPi):
	k = len(MAB)
	Pi_n = PosteriorDistrib(k)
	reco = []
	for t in range(T):
		# first estimate each alpha_i using the algorithm 2
		# the posterior distrib is given by Pi_n as in TTTS
		theta = np.zeros((M, k))
		for j in range(M):			
			theta[j]=Pi_n.sample()
		S = np.argmax(theta, axis=1)

		alpha = np.zeros(k)

		# we use np.unique to compute alpha faster
		unique, counts = np.unique(S, return_counts=True)

		for i in range(len(unique)):
			alpha[unique[i]] = counts[i]/M

		# choose I and J
		I = np.argmax(alpha)
		alpha[I] = -np.inf
		J = np.argmax(alpha)

		# plays I or J according a bernoulli variable
		B = np.random.binomial(1, beta)
		if B==1:
			# play I
			r = MAB[I].sample()
			Pi_n.update(r, I)

		else:
			# play J
			r = MAB[J].sample()
			Pi_n.update(r, J)

		# we recommend the best arm in posterior mean
		posterior_best = np.argmax(Pi_n.mu)
		reco.append([posterior_best])

	return reco

def TTVS(MAB, T=10000, M=20, beta=0.5, PosteriorDistrib=GaussianPi):
	# we use TTVS with the identity as utility function
	k = len(MAB)
	Pi_n = PosteriorDistrib(k)
	reco = []
	for t in range(T):
		# first estimate each V_i using the algorithm 2
		# the posterior distrib is given by Pi_n as in TTTS
		theta = np.zeros((M, k))
		for j in range(M):			
			theta[j]=Pi_n.sample()
		S = np.argmax(theta, axis=1)

		V = np.zeros(k)
		for j in range(M):
			V[S[j]] += theta[j, S[j]]
			# we look for the max for i different than j
			theta[j, S[j]] = -np.inf
			V[S[j]] -= np.max(theta[j])
		V = V/M 

		# choose I and J
		I = np.argmax(V)

		# plays I or J according a bernoulli variable
		B = np.random.binomial(1, beta)
		if B==1:
			# play I
			r = MAB[I].sample()
			Pi_n.update(r, I)

		else:			
			V[I] = -np.inf
			J = np.argmax(V)
			# play J
			r = MAB[J].sample()
			Pi_n.update(r, J)

		# we recommend the best arm in posterior mean
		posterior_best = np.argmax(Pi_n.mu)
		reco.append([posterior_best])

	return reco