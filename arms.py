import numpy as np
from random import shuffle

class AbstractArm(object):
    def __init__(self, mean, variance, random_state):
        """
        Args:
            mean: expectation of the arm
            variance: variance of the arm
            random_state (int): seed to make experiments reproducible
        """
        self.mean = mean
        self.variance = variance

        self.local_random = np.random.RandomState(random_state)

    def sample(self):
        pass


class ArmGaussian(AbstractArm):
    
    def __init__(self, mu=0, sigma2=1/4, random_state=0, clipping=False):
        """
        Gaussian arm
        Args:
             mu (float): mean parameter
             sigma2 (float): variance
             random_state (int): seed to make experiments reproducible
             clipping : if set to True, then the arms can only output in [0, 1]
        """
        self.mu = mu
        self.sigma2 = sigma2
        self.clipping = clipping
        super(ArmGaussian, self).__init__(mean=mu,
                                           variance=sigma2,
                                           random_state=random_state)

    def sample(self):
        s = self.local_random.normal(self.mu, np.sqrt(self.sigma2))
        if self.clipping:
            return np.clip(s, 0, 1)
        else:
            return s


def ToyMABLinear(n):
	# returns a linear instance ToyMAB of n arms
	# as explained in the paper [1]

    # we shuffle it so that the order of the arms is random
    # it avoids biased results in some algorithms
    r = np.random.randint(21578963, size=n)
    MAB = [ArmGaussian(0.9*(n-i)/(n-1), 1/4, r[i]) for i in range(n)]
    shuffle(MAB)
    return MAB
    

def ToyMABpoly(n):
	# returns a polynomial instance ToyMAB of n arms
	# as explained in the paper [1]

    r = np.random.randint(215789623, size=n)
    MAB = [ArmGaussian(mu=0.9, sigma2=1/4, random_state=r[0])]
    for i in range(2, n+1):
        MAB.append(ArmGaussian(0.9*(1-np.sqrt(i/n)), 1/4, random_state=r[i-1]))

    shuffle(MAB)
    return MAB

