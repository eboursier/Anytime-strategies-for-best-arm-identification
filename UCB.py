
import numpy as np

def UCB_MAP(MAB, T=10000, alpha=2):
    
    bests = []
    K = len(MAB)
    mu_hat = np.zeros(K) 
    N = np.zeros(K)
    
    for t in range (K):
        arm = MAB[t]
        a = arm.sample()
        N[t]+=1
        mu_hat[t] = a
        bests.append([t])
    for t in range(K, T):
        ind = np.argmax(mu_hat + np.sqrt(alpha*np.log(t)/(N)))
        arm = MAB[ind] 
        a = arm.sample()
        N[ind]+=1
        mu_hat[ind] = (mu_hat[ind]*N[ind]+a)/(N[ind]+1)    
        most_drawn = np.argmax(N)
        bests.append([most_drawn])
    return bests



