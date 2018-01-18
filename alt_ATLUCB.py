import numpy as np


def KL_distance(p, q):
	return(p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q)))

def KL_level_low(p, level, tol2=16):
	"""
	return q in [0, p] such that KL_distance(p, q) = level
	We do a dichotomia
	"""
	if p < 0:
		return p


	up = p
	low = 0
	for _ in range(tol2):
		q = (up+low)/2
		val = KL_distance(p, q)
		if val > level:
			low = q
		else:
			up = q

	return q

def KL_level_up(p, level, tol2=16):
	"""
	return q in [0, p] such that KL_distance(p, q) = level
	We do a dichotomia
	"""
	if p>1:
		return p

	up = 1
	low = p
	for _ in range(tol2):
		q = (up+low)/2
		val = KL_distance(p, q)
		if val > level:
			up = q
		else:
			low = q

	return q

def UL_alternative(MAB_means, n_samples, t, delta, J):

	"""
	Return the confidence bounds U and L introduced in the paper Information Complexity in Bandit Subset Selection
	The upper bound U[a] of the arm a is : sup{ 1> q > MAB_means[a] / MAB_means[a]log(MAB_means[a]/q) + (1-MAB_means[a])log((1-MAB_means[a])/(1-q)) < log(5*n*t**4/(4*delta))/n_samples[a]}
	The lower bound L[a] of the arm a is : inf{ 0 < q < MAB_means[a] / ... }
	if n_samples is 0, then 0 and 1 are automatically the upper and lower bounds
	Also, we then set U[J] = -np.inf and we only return L[J] so that we choose the arms in J for L and in not(J) for U

	Inputs:
	MAB_means: the empirical means
	n_samples: the number of pulls for each arm
	t: the time step
	delta: the confidence level
	best: the empirical top-m arms
	"""
	n = len(MAB_means)
	U = np.zeros(n)
	L = np.zeros(len(J))

	# create U
	for a in range(n):
		if a in J:
			U[a] = - np.inf
		elif n_samples[a] == 0:
			U[a] = 1
		else:
			level =  np.log((5*n*t**4)/(4*delta))/n_samples[a]
			q_u = KL_level_up(MAB_means[a], level)
			U[a] = q_u

	# create L
	for i, a in enumerate(J):
		if n_samples[a] == 0:
			L[i] = 0
		else:
			level = np.log((5*n*t**4)/(4*delta))/n_samples[a]
			q_l = KL_level_low(MAB_means[a], level)
			L[i] = q_l

	return U, L


def alternative_ATLUCB(MAB, m=1, delta1=0.01, alpha=0.5, epsilon=0, T=1000):

	#output a list of recommandations at each time step t

	t = 1
	S = 1

	n = len(MAB)
	MAB_means = np.zeros(n)
	n_samples = np.zeros(n)
	best = np.argpartition(MAB_means, -m)[-m:]
	
	recommandations = []
	delta_S = delta1

	term = False
	# here we stop when we reach an horizon
	# because we draw two arms at each stage, we divide the actual horizon by 2
	while (t<=T):
		if term:
			term_S = True

			# J is the top m arms of MAB
			J = best

			# compute the new S
			while term_S:
				S += 1
				delta_S *= alpha


				U, L = UL_alternative(MAB_means, n_samples, t, delta_S, J=J)
				# check if we have terminaison
				gap = np.max(U) - np.min(L)
				term_S = (gap < epsilon)

		else:
			if S == 1:
				# J is the top m arms of MAB
				try:
					J = best
				except:
					J = np.argpartition(MAB_means, -m)[-m:]

		# determine h* and l*
		U, L = UL_alternative(MAB_means, n_samples, t, delta_S, J=best)

		l = np.argmax(U)
		h = best[np.argmin(L)]

		# draw once h* and l*
		r_h = MAB[h].sample()
		r_l = MAB[l].sample()

		# update MAB_means and n_samples
		MAB_means[h] = MAB_means[h]*n_samples[h]/(n_samples[h]+1) + r_h/(n_samples[h]+1)
		MAB_means[l] = MAB_means[l]*n_samples[l]/(n_samples[l]+1) + r_l/(n_samples[l]+1)

		n_samples[h] += 1
		n_samples[l] += 1

		# update t, term and the recommandations list
		t += 2

		best = np.argpartition(MAB_means, -m)[-m:]
		#check if we have terminaison
		U, L = UL_alternative(MAB_means, n_samples, t, delta_S, J=best)

		# check if we have terminaison
		gap = np.max(U) - np.min(L)
		term = (gap < epsilon)


		# we add twice J in the recommandations, because ATLUCB draws 2 arms per round
		recommandations.append(J)
		recommandations.append(J)

	return recommandations