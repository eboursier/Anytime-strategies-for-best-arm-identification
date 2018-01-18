import numpy as np

def deviation_beta(n, u, t, delta, k1=5/4):
	# return beta(u, t, delta) as described in the paper [1]
	return np.sqrt(np.divide(np.log(k1*n*(t**4)/delta),2*u))

def ATLUCB(MAB, m=1, delta1=0.01, alpha=0.5, epsilon=0, T=1000, deviation=deviation_beta):

	#output a list of recommandations at each time step t

	t = 1
	S = 1

	n = len(MAB)
	MAB_means = np.zeros(n)
	n_samples = np.zeros(n)

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
			U = np.copy(MAB_means)
			L = np.copy(MAB_means[J])

			U[J] = -np.inf

			# compute the new S
			while term_S:
				S += 1
				delta_S *= alpha
				dev = deviation(n, n_samples, t, delta_S)

				# check if we have terminaison
				gap = np.max(U+dev) - np.min(L-dev[J])
				term_S = (gap < epsilon)

		else:
			if S == 1:
				# J is the top m arms of MAB
				try:
					J = best
				except:
					J = np.argpartition(MAB_means, -m)[-m:]

		if t==1:
			best = J

		# determine h* and l*
		dev = deviation(n, n_samples, t, delta_S)
		U = MAB_means + dev
		L = MAB_means[best] - dev[best]

		U[best] = -np.inf

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

		# check if we have term
		dev = deviation(n, n_samples, t, delta_S)
		U = MAB_means + dev
		L = MAB_means[best] - dev[best]

		U[best] = -np.inf

		# check if we have terminaison
		gap = np.max(U) - np.min(L)
		term = (gap < epsilon)


		# we add twice J in the recommandations, because ATLUCB draws 2 arms per round
		recommandations.append(J)
		recommandations.append(J)

	return recommandations
