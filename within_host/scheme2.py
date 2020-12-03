import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

########################
# Manual adjustments to simulation

# Parameters
Parameters =   {"Num_mic"   :27, # number of microbes: should agree with input microbe array if running for given microbes
                "K_M"       :1, # total load of microbes
                "Horiz_T"   :0, # Rate of colonization
                "Env_update":0, # Rate of update of Environment microbe pool by host
                "Heps"      :0, # Host environment value
                "T"         :1000, # Length of simulation
                "s_Meps"    :1, # magnitude of effect of microbes on microbe environment
                "s_Mphi"    :1, # magnitude of effect of microbes on host phenotype
                "sigmaM"    :0.5, # controls magnitude of microbe selection
                "sigmaH"    :0.5  # controls magnitude of host selection
                }

# Microbes
# Manually input microbe trait characteristics here - will be used if simulation sun for given microbes
Microbes = np.array([[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1], # Microbe phenotype
                    [1,0,-1,1,0,-1,1,0,-1,1,0,-1,1,0,-1,1,0,-1,1,0,-1,1,0,-1,1,0,-1],  # effect on microbe environment
                    [1,1,1,0,0,0,-1,-1,-1,1,1,1,0,0,0,-1,-1,-1,1,1,1,0,0,0,-1,-1,-1]]) # effect on host phenotype

# Host
Host = [0,0]   # Host trait

# Number of trials
TT = 10000
#########################

def run_scheme2_singlehost(Parameters, Microbes, Host, init_A):

    # Environmental acquisitions and Host phenotype not considered. Can be changed.

    Phi_M = Microbes[0]
    E_MM = Microbes[1]
    E_MH = Microbes[2]
    K_M = Parameters["K_M"]
    N = Parameters["Num_mic"]

    s_Meps = Parameters["s_Meps"]
    s_Heps = 1-s_Meps
    s_Mphi = Parameters["s_Mphi"]
    s_Hphi = 1-s_Mphi

    sigmaM = Parameters["sigmaM"]
    sigmaH = Parameters["sigmaH"]

    E_HM = Host[0]
    E_HH = Host[1]

    Heps = Parameters["Heps"]

    # Environment pool
    #Env = np.full(N, K_M / N)   # initialize environment pool abundances

    #HT = Parameters["Horiz_T"]
    #Env_upd = Parameters["Env_update"]

    time_steps = Parameters["T"]
    A = init_A

    Abundance_Output = []     # time series of microbe abundances
    Parameter_Output = []

    for t in range(time_steps):
        eps = s_Meps * sum(A * E_MM) + s_Heps * E_HM # microbe environment value
        phi = s_Mphi * sum(A * E_MH) + s_Hphi * E_HH # host phenotype value

        # Microbe fitness
        R_M = np.exp(-(1 / (2 * sigmaM ** 2)) * (eps - Phi_M) ** 2)
        # Microbe growth
        A = A * (1 + R_M)
        # Normalize population
        A = A / sum(A)
        # Colonization
        #A = A + HT*Env
        #A = A / sum(A)

        # Host fitness
        R_H = np.exp(-(1 / (2 * sigmaH ** 2)) * (Heps - phi) ** 2) / math.sqrt(2 * math.pi) / 0.4

        Abundance_Output.append(A)
        Parameter_Output.append(np.array([eps,phi,R_H]))

    return (Abundance_Output, Parameter_Output)


def stat_dist_given_microbes(Parameters, Microbes, Host, TT):  # Obtain microbe abundances at 1000th time-step for TT many trials with
                                                # random initial abundances
    N = Parameters["Num_mic"]
    stat_dist = pd.DataFrame(columns=list(str(i) for i in range(N)))
    stat_par = pd.DataFrame(columns=['eps', 'phi', 'R_H'])

    for T in range(TT):
        init_A = np.random.rand(N)              # Generate random initial abundance
        init_A = init_A/sum(init_A)

        dist, par = run_scheme2_singlehost(Parameters, Microbes, Host, init_A)

        stat_par = stat_par.append(dict(eps=par[-1][0], phi=par[-1][1], R_H=par[-1][2]), ignore_index=True)
        stat_dist[T] = dist[-1]
        print(T)
    return(stat_dist, stat_par)

def stat_dist_random_microbes(Parameters, Host,TT):  # Obtain microbe abundances at 1000th time-step for TT many trials with
                                               # random initial abundances
    N = Parameters["Num_mic"]
    stat_dist = pd.DataFrame(columns=list(str(i) for i in range(N)))
    stat_par = pd.DataFrame(columns=['eps', 'phi', 'R_H'])

    for T in range(TT):
        init_A = np.random.rand(N)  # Level 1 of randomness: Initial abundances
        init_A = init_A / sum(init_A)

        Microbes = np.random.rand(3, N) * 2 - 1  # Level 2 of randomization: Random microbe trait sample | Optional
                                                                             # can read as non-random input instead
        Microbes[0].sort()  # Order effect on self. Replace 0 with 1 to order wrt effect on other microbes
                                                  # Replace 0 with 2 to order wrt effect on host fitness

        dist, par = run_scheme2_singlehost(Parameters, Microbes, Host, init_A)

        stat_par = stat_par.append(dict(eps=par[-1][0], phi=par[-1][1], R_H=par[-1][2]), ignore_index=True)
        stat_dist[T] = dist[-1]
        print(T)
    return (stat_dist, stat_par)

dist, par = stat_dist_given_microbes(Parameters, Microbes, Host, TT)

sns.set_theme()

fig, axes = plt.subplots(nrows=1,ncols=2)

sns.stripplot(data = dist.T, ax=axes[0])
axes[0].set_xlabel("Microbe index")
axes[0].set_ylabel("Microbe abundance at t={}".format(Parameters["T"]))
sns.boxplot(data= par, ax=axes[1])
sns.stripplot(data= par, ax=axes[1], alpha=0.25)
fig.tight_layout()

plt.show()