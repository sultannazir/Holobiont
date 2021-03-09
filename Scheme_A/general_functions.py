import numpy as np

def random_initialize(M,N):
    # Microbe interaction matrix from N microbes to N other microbes + 1 host locus (N x N+1)
    I = np.array([np.random.choice(np.array([-1, 0, 1]), N + 1) for i in range(N)])
    # Initial Host fitness effects from M host loci across M hosts to N microbes and itself (M x N+1)
    RH = np.random.normal(0,0.1,(M, N + 1))
    # Initial microbe abundances in non-zero integers (M x N)
    A = np.full((M, N),1)
    # Initial host controlled vertical transmission values from M hosts and N microbes within each host (M x N)
    T = np.full((M,N),0)
    # Initial environment microbe pool abundance that sums up to 1
    Env = np.full(N, 1 / N)
    return(I, RH, T, A, Env)

def calc_rowwise_norm(A): # To find relative microbe abundances in each host
    loads = np.sum(A, axis=1)
    loads[loads==0] = 1         # avoid microbe-free hosts
    A0 = A / loads[:,np.newaxis]
    return(A0)

def find_R(I, A, RH, rM, s):
    A0 = calc_rowwise_norm(A)
    RM = np.matmul(A0,I)  # net microbe effect on fitness as per unit fitness effect (I) multiplied by abundance (A)
    R = rM + s*RH + (1-s)*RM  # net fitness as sum of host effect and microbe effect to intrinsic fitness
    # extract and normalize microbe fitnesses in each host
    micR = R[:,:-1]/np.clip(np.sum(R[:,:-1],axis=1)[:,np.newaxis], a_min=0.0001, a_max=None)
    # extract and normalize host fitesses
    hostR = R[:,-1]/np.clip(np.sum(R[:,-1]), a_min=0.0001, a_max=None)
    return(micR, hostR)

def update_microbes(A, micR, Env, c, K):

    A0 = calc_rowwise_norm(A)
    AR = np.multiply(A0, micR)  # net growth rate = abundance x net fitness
    normAR = calc_rowwise_norm(AR)  # normalized growth rate

    P1 = A/K             # probability of death
    P2 = (1-c)*normAR    # probability of proliferation
    P3 = c*Env           # probability of colonization

    PA = np.multiply(P1, 1-P2-P3)
    PB = PA + np.multiply(1 - P1, P2+P3)

    Probs = np.random.rand(A.shape[0],A.shape[1])

    Aadd = np.zeros(A.shape)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if Probs[i,j]<= PA[i,j]:
                Aadd[i,j] = -1
            elif PA[i,j] < Probs[i,j]<= PB[i,j]:
                Aadd[i,j] = 1
            else:
                continue

    return(A+Aadd)


def update_env(Env, A, Env_upd):
    A0 = calc_rowwise_norm(A)    # average abundance of microbes in the host population
    A0 = np.mean(A0, axis=0)
    Env = (1-Env_upd)*Env + Env_upd*A0  # Env_upd fraction of environment filled by average host abundance value
    Env = Env/sum(Env)  # normalize again to account for a case of microbe-free host population
    return(Env)

def stepfunc(x):
    x = np.ceil(np.clip(x,a_min=0, a_max=1))
    return(x)

def update_host(A, R, RH, T, mu, d, seed):
    rand = np.random.rand()
    IDhost = np.arange(len(R)) # index hosts from 0 to M-1

    if rand<=d:
        Bcumvec = R.cumsum() # cumulative fitness to get growth propensity vector
        Dcumvec = (1-R).cumsum() # cumulative 1-fitness for death propensity vector
        randnum = np.random.rand()
        IDbirth = IDhost[(Bcumvec > randnum)][0] # choose index of host selected to reproduce
        IDdeath = IDhost[(Dcumvec > randnum*Dcumvec[-1])][0]
        #IDdeath = np.random.choice(IDhost) # choose index of host which dies

        # update offspring host control values
        RH[IDdeath] = np.clip(RH[IDbirth] + np.random.normal(0, 0, len(RH[0])), a_min=-1, a_max=1)
        # update offspring host-controlled transmission values
        T[IDdeath] = np.clip(T[IDbirth] + np.random.normal(0, 0, len(RH[0])-1), a_min=-1, a_max=1)

        # initalize offspring microbiome
        randtrans = np.random.rand(len(A[0]))

        A0 = calc_rowwise_norm(A)
        for i in range(len(A[0])):
            if T[IDbirth][i]*A0[IDbirth][i]>randtrans[i]:
                # random number says true for transmission
                # multiply with step func to check presence of microbe in the parent
                # If both satisfied, seed microbe in the offspring
                A[IDdeath][i] = seed
            else:
                A[IDdeath][i] = 0

    return(A, RH, T)

#################################################################################################################
# Functions to run scheme

# All initial matrices are given
def run_schemeA_giveninit(I, RH, T, A, Env, Parameters):
    d = Parameters['death_rate']
    s = Parameters['host_weight']
    mu = Parameters['mutation']

    r = Parameters['mic_intR']
    K = Parameters['mic_capacity']

    c = Parameters['col_rate']
    seed = Parameters['seed']
    EU = Parameters['Env_update']

    TT = Parameters['sim_time']
    bins = Parameters['num_bins']

    Abundances = []
    Transmissions = []
    Host_control = []
    for i in range(len(I)):
        Abundances.append([])
        Transmissions.append([])
        Host_control.append([])

    for t in range(TT):
        A0 = calc_rowwise_norm(A)
        for i in range(len(I)):
            # Give distribution of abundance values of each microbe at time t
            Abundances[i].append(np.flip(np.histogram(A0[:, i], bins=np.linspace(0, 1, bins))[0]))
            # Give distribution of transmission values on each microbe at time t
            Transmissions[i].append(np.flip(np.histogram(T[:, i], bins=np.linspace(0, 1, bins))[0]))
            # Give distribution of host control values on each microbe at time t
            Host_control[i].append(np.flip(np.histogram(RH[:, i], bins=np.linspace(-1, 1, bins))[0]))

        Env = update_env(Env, A, EU)
        micR, hostR = find_R(I, A, RH, r, s)
        A = update_microbes(A, micR, Env, c, K)
        A, RH, T = update_host(A, hostR, RH, T, mu, d, seed)
        print(t)

    return (Abundances, Transmissions, Host_control)


# For fully random initial conditions where initial matrix values are all from random_initialize function output
def run_schemeA_randominit(Parameters):
    N = Parameters['Num_mic']
    M = Parameters['Num_host']

    I, RH, T, A, Env = random_initialize(M, N)

    AB, T, RH = run_schemeA_giveninit(I, RH, T, A, Env, Parameters)

    return(AB, T, RH)


# All matrices except I are output of random_init function. I is given manually
def run_schemeA_givenI(I,Parameters):
    N = Parameters['Num_mic']
    M = Parameters['Num_host']

    X, RH, T, A, Env = random_initialize(M, N)

    AB, T, RH = run_schemeA_giveninit(I, RH, T, A, Env, Parameters)

    return (AB, T, RH)

def run_schemeA_singlehost(I, RH, A, Env, Parameters):
    s = Parameters['host_weight']

    r = Parameters['mic_intR']
    K = Parameters['mic_capacity']

    c = Parameters['col_rate']

    TT = Parameters['sim_time']

    Abundance = []

    for t in range(TT):
        Abundance.append(A[0])
        micR, hostR = find_R(I, A, RH, r, s)
        A = update_microbes(A, micR, Env, c, K)

    Abundance.append(A[0])
    return(Abundance)

def run_schemeA_frac_singlehost(I, RH, A, Env, Parameters):
    s = Parameters['host_weight']

    r = Parameters['mic_intR']
    K = Parameters['mic_capacity']

    c = Parameters['col_rate']

    TT = Parameters['sim_time']

    Abundance = []

    for t in range(TT):
        A0 = calc_rowwise_norm(A)
        Abundance.append(A0[0])
        micR, hostR = find_R(I, A, RH, r, s)
        A = update_microbes(A, micR, Env, c, K)

    return(Abundance)

def run_schemeA_giveninit_getmean(I, RH, T, A, Env, Parameters):
    d = Parameters['death_rate']
    s = Parameters['host_weight']
    mu = Parameters['mutation']

    r = Parameters['mic_intR']
    K = Parameters['mic_capacity']

    c = Parameters['col_rate']
    seed = Parameters['seed']
    EU = Parameters['Env_update']

    TT = Parameters['sim_time']

    Abundances = []
    Trans = []
    for i in range(len(I)):
        Abundances.append([])
        Trans.append([])

    for t in range(TT):
        A0 = calc_rowwise_norm(A)
        for i in range(len(I)):
            # Give mean of relative abundance values of each microbe at time t
            Abundances[i].append(np.mean(A0,axis=0)[i])
            Trans[i].append(np.mean(T, axis=0)[i])

        Env = update_env(Env, A, EU)
        micR, hostR = find_R(I, A, RH, r, s)
        A = update_microbes(A, micR, Env, c, K)
        A, RH, T = update_host(A, hostR, RH, T, mu, d, seed)



    return (Abundances, Trans)

def run_schemeA_giveninit_track(I, RH, T, A, Env, Parameters):
    d = Parameters['death_rate']
    s = Parameters['host_weight']
    mu = Parameters['mutation']

    r = Parameters['mic_intR']
    K = Parameters['mic_capacity']

    c = Parameters['col_rate']
    seed = Parameters['seed']
    EU = Parameters['Env_update']

    TT = Parameters['sim_time']

    Abundances = []
    Trans = []
    for i in range(len(I)):
        Abundances.append([])
        Trans.append([])

    for t in range(TT):
        A0 = calc_rowwise_norm(A)
        for i in range(len(I)):
            # Give mean of relative abundance values of each microbe at time t
            Abundances[i].append(A0[:,i])
            Trans[i].append(np.mean(T, axis=0)[i])

        Env = update_env(Env, A, EU)
        micR, hostR = find_R(I, A, RH, r, s)
        A = update_microbes(A, micR, Env, c, K)
        A, RH, T = update_host(A, hostR, RH, T, mu, d, seed)



    return (Abundances, Trans)
