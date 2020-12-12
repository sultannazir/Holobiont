import numpy as np

def random_initialize(M,N):
    I = np.array([np.random.choice(np.array([-1, 0, 1]), N + 1) for i in range(N)])
    RH = np.random.normal(0,0.1,(M, N + 1))
    A = np.zeros((M, N))
    T = np.zeros((M,N))
    Env = np.full(N, 1 / N)
    return(I, RH, T, A, Env)

def calc_rowwise_norm(A):
    loads = np.sum(A, axis=1)
    loads[loads==0] = 1         # set non-zero loads
    A0 = A / loads[:,np.newaxis]
    return(A0)

def find_R(I, A, RH, rM, s):
    A0 = calc_rowwise_norm(A)
    RM = np.matmul(A0,I) # net microbe effect on fitness
    R = rM + s*RH + (1-s)*RM # net fitness
    micR = R[:,:-1]/np.sum(R[:,:-1],axis=1)[:,np.newaxis]
    hostR = R[:,-1]/np.sum(R[:,-1])
    return(micR, hostR)

def update_microbes(A, R, Env, c, K):

    A0 = calc_rowwise_norm(A)
    AR = np.multiply(A0, R)  # net growth rate
    normAR = calc_rowwise_norm(AR)

    P1 = A/K    # probability of death
    P2 = (1-c)*normAR    # probability of proliferation
    P3 = c*Env  # probability of colonization

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
    Env = (1-Env_upd)*Env + Env_upd*A0
    Env = Env/sum(Env)
    return(Env)

def stepfunc(x):
    x = np.ceil(np.clip(x,a_min=0, a_max=1))
    return(x)

def update_host(A, R, RH, T, mu, d):
    rand = np.random.rand()
    IDhost = np.arange(len(R)) # index hosts

    if rand<=d:
        cumvec = R.cumsum() # cumulative prop using fitness propensities
        randnum = np.random.rand()
        IDbirth = IDhost[(cumvec > randnum)][0] # choose index of host selected to reproduce

        IDdeath = np.random.choice(IDhost) # choose index of host which dies

        RH[IDdeath] = np.clip(RH[IDbirth] + np.random.normal(0, mu, len(RH[0])), a_min=-1, a_max=1)
        T[IDdeath] = np.clip(T[IDbirth] + np.random.normal(0, mu, len(RH[0])-1), a_min=-1, a_max=1)

        randtrans = np.random.rand(len(A[0]))
        for i in range(len(A[0])):
            if T[IDbirth][i]*stepfunc(A[IDbirth][i])<randtrans[i]:
                A[IDdeath][i] = 1
            else:
                A[IDdeath][i] = 0

    return(A, RH, T)


def run_schemeA_randominit(Parameters):
    N = Parameters['Num_mic']
    M = Parameters['Num_host']

    d = Parameters['death_rate']
    s = Parameters['host_weight']
    mu = Parameters['mutation']

    r = Parameters['mic_intR']
    K = Parameters['mic_capacity']

    c = Parameters['col_rate']
    EU = Parameters['Env_update']

    TT = Parameters['sim_time']

    I, RH, T, A, Env = random_initialize(M, N)

    Output = []
    for t in range(TT):
        Output.append(np.mean(A, axis=0))
        Env = update_env(Env, A, EU)
        micR, hostR = find_R(I, A, RH, r, s)
        A = update_microbes(A, micR, Env, c, K)
        A, RH, T = update_host(A, hostR, RH, T, mu, d)
        print(t)
    Output.append(np.mean(A, axis=0))
    return(Output)

def run_schemeA_givenI(I,Parameters):
    N = Parameters['Num_mic']
    M = Parameters['Num_host']

    d = Parameters['death_rate']
    s = Parameters['host_weight']
    mu = Parameters['mutation']

    r = Parameters['mic_intR']
    K = Parameters['mic_capacity']

    c = Parameters['col_rate']
    EU = Parameters['Env_update']

    TT = Parameters['sim_time']
    bins = Parameters['num_bins']

    X, RH, T, A, Env = random_initialize(M, N)

    Abundances= []
    Transmissions = []
    Host_control = []
    for i in range(len(I)):
        Abundances.append([])
        Transmissions.append([])
        Host_control.append([])

    for t in range(TT):

        for i in range(len(I)):
            Abundances[i].append(np.flip(np.histogram(A[:,i], bins=np.linspace(0,K,bins))[0]))
            Transmissions[i].append(np.flip(np.histogram(T[:,i], bins=np.linspace(0,1,bins))[0]))
            Host_control[i].append(np.flip(np.histogram(RH[:, i], bins=np.linspace(-1, 1, bins))[0]))

        Env = update_env(Env, A, EU)
        micR, hostR = find_R(I, A, RH, r, s)
        A = update_microbes(A, micR, Env, c, K)
        A, RH, T = update_host(A, hostR, RH, T, mu, d)
        print(t)

    return(Abundances,Transmissions,Host_control)

def run_schemeA_giveninit(I,RH, T, A, Env, Parameters):

    Output = []

    d = Parameters['death_rate']
    s = Parameters['host_weight']
    mu = Parameters['mutation']

    r = Parameters['mic_intR']
    K = Parameters['mic_capacity']

    c = Parameters['col_rate']
    EU = Parameters['Env_update']

    TT = Parameters['sim_time']

    for t in range(TT):
        Output.append(np.mean(A,axis=0))
        Env = update_env(Env, A, EU)
        micR, hostR = find_R(I, A, RH, r, s)
        A = update_microbes(A, micR, Env, c, K)
        A, RH, T = update_host(A, hostR, RH, T, mu, d)

        print(t)
    Output.append(np.mean(A,axis=0))
    return(Output)
