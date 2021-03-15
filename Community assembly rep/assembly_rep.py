import numpy as np
import general_functions as gf

P = 0.5 # ratio interacting
c = 0.1 # colonizaztion rate
T = 5000 # simulation time
N = 5 # number of microbes
trials = 100 # number of random trials
bins = 5 # number of values of percpos
K = 1000 # carrying capacity

percpos = 0
Env = np.full(N,1/N)

hist = []
while percpos <= 1:
    As = []
    for i in range(trials):
        S = np.random.uniform(0,1,(N,N))
        Ipos = S < percpos*P
        Ineg = S > percpos + (1-P)*(1 - percpos)
        Iint = Ipos.astype(int) - Ineg.astype(int)
        np.fill_diagonal(Iint,0)

        I = Iint * np.clip(abs(np.random.normal(0,1,(N,N))), a_min=0, a_max = 1)

        A = np.full(N,0)

        for t in range(T):

            micR = gf.find_R_nohost(I,A,1)
            A = gf.update_microbes_new(A, micR, Env, c, K)
        assembled = np.sum((A>100).astype(int))
        As.append(assembled)
    hist.append(sum(As)/trials)
    percpos += 1/bins

print(hist)