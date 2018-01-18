import numpy as np

def DSH(MAB, T = 5000, m=1):
    n = len(MAB)
    s = 0
    bests = []
    pulls_count = 0
    rew_means = np.zeros(n)
    n_draws = np.zeros(n)
    
    while(pulls_count<T):
        s+=1
        T0 = n*2**(s-1)
        S_r = np.arange(n)
        n_arms = n
        
        for r in range((int)(np.ceil(np.log2(n)))):
            
            n_pull = (int)(np.floor(T0/(n_arms*np.ceil(np.log2(n)))))
            for i in range(n_arms):
                j = S_r[i]
                arm = MAB[j]                
                for t in range(n_pull):
                    a = arm.sample()
                    rew_means[j] = (rew_means[j]*n_draws[j]+a)/(n_draws[j]+1)
                    n_draws[j] += 1
                    pulls_count += 1
                    bests.append(np.argpartition(rew_means, -m)[-m:])
                    
            
            sort_index = np.argsort(-rew_means[S_r])
            
            n_kept = (int)(np.ceil(n_arms/2))
            S_r = S_r[sort_index[:n_kept]]
            n_arms = len(S_r)
        
    return bests[:T]

def DSH_Naive(MAB, T = 5000, m=1):
    n = len(MAB)
    s = 0
    bests = []
    previous_stage_best = np.random.choice(n, size=m, replace = False)
    pulls_count = 0
    rew_means = np.zeros(n)
    n_draws = np.zeros(n)
    while(pulls_count<T):
        s+=1
        T0 = n*2**(s-1)
        S_r = np.arange(n)
        n_arms = n
        
        for r in range((int)(np.ceil(np.log2(n)))):
            
            n_pull = (int)(np.floor(T0/(n_arms*np.ceil(np.log2(n)))))
            for i in range(n_arms):
                j = S_r[i]
                arm = MAB[j]                
                for t in range(n_pull):
                    a = arm.sample()
                    rew_means[j] = (rew_means[j]*n_draws[j]+a)/(n_draws[j]+1)
                    n_draws[j] += 1
                    pulls_count += 1
                    bests.append(previous_stage_best)
                    
            
            sort_index = np.argsort(-rew_means[S_r])
            
            n_kept = (int)(np.ceil(n_arms/2))
            S_r = S_r[sort_index[:n_kept]]
            n_arms = len(S_r) 
        previous_stage_best = np.argpartition(rew_means, -m)[-m:]
    return bests[:T]


def Uniform(MAB, T, m=1):
    n = len(MAB)
    reco = []
    rew_means = np.zeros(n)
    n_draws = np.zeros(n)
    for t in range(T):
        a = np.argmin(n_draws)
        r = MAB[a].sample()
        rew_means[a] = (rew_means[a]*n_draws[a]+r)/(n_draws[a]+1)
        n_draws[a] += 1
        J = np.argpartition(rew_means, -m)[-m:]
        reco.append(J)
    return reco

def DSAR_Naive(MAB, T = 5000, m=1):
    
    K = len(MAB)
    logbar = np.sum([1/i for i in range(2,K+1)]) + 1/2
    A = np.arange(K)
    n_k = np.zeros(K)
    rew_means = np.zeros(K)
    n_draws = np.zeros(K)
    bests = []
    previous_stage_best = np.random.choice(K, size=m, replace = False).tolist()
    s = 0
    pulls_count = 0
    while (pulls_count<T):
        s += 1
        n = m
        T0 = K*2**(s-1)
        deact = []
        for k in range(1, K):
            n_k[k] = (np.ceil((T0 - K)/(logbar*(K+1-k))))
            n_pull = (int)(n_k[k]-n_k[k-1])
            C = np.setdiff1d(A, deact)
            for i in C:
                arm = MAB[i]
                for t in range(n_pull):
                    r = arm.sample()
                    pulls_count += 1
                    rew_means[i] = (rew_means[i]*n_draws[i]+r)/(n_draws[i]+1)
                    n_draws[i] += 1
                    bests.append(previous_stage_best)
            mu = np.copy(rew_means)
            mu[deact] = -np.inf
            sigma = np.argsort(-mu)
            delta = -np.inf*np.ones(len(A))
            for r in range(len(C)):
                if (r<=n): delta[sigma[r]] = mu[sigma[r]]-mu[sigma[n]]
                else: delta[sigma[r]] = mu[sigma[n-1]]-mu[sigma[r]]
        
            i_k = np.argmax(delta)
            deact.append(i_k)
            
            if (mu[i_k] > mu[sigma[n]]):
                n -= 1
        previous_stage_best = np.argpartition(rew_means, -m)[-m:]
            
    return bests[:T]
    

def DSAR(MAB, T = 5000, m=1):
    
    K = len(MAB)
    logbar = np.sum([1/i for i in range(2,K+1)]) + 1/2
    A = np.arange(K)
    n_k = np.zeros(K)
    rew_means = np.zeros(K)
    n_draws = np.zeros(K)
    bests = []
    s=0
    pulls_count = 0
    while (pulls_count<T):
        s += 1
        n = m
        T0 = K*2**(s-1)
        deact = []
        for k in range(1, K):
            n_k[k] = (np.ceil((T0 - K)/(logbar*(K+1-k))))
            n_pull = (int)(n_k[k]-n_k[k-1])
            C = np.setdiff1d(A, deact)
            for i in C:
                arm = MAB[i]
                for t in range(n_pull):
                    r = arm.sample()
                    pulls_count += 1
                    rew_means[i] = (rew_means[i]*n_draws[i]+r)/(n_draws[i]+1)
                    n_draws[i] += 1
                    bests.append(np.argpartition(rew_means, -m)[-m:])
            mu = np.copy(rew_means)
            mu[deact] = -np.inf
            sigma = np.argsort(-mu)
            delta = -np.inf*np.ones(len(A))
            for r in range(len(C)):
                if (r<=n): delta[sigma[r]] = mu[sigma[r]]-mu[sigma[n]]
                else: delta[sigma[r]] = mu[sigma[n-1]]-mu[sigma[r]]
        
            i_k = np.argmax(delta)
            deact.append(i_k)
            
            if (mu[i_k] > mu[sigma[n]]):
                n -= 1
        
            
    return bests[:T]
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    