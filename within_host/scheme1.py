import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

########################
# Manual adjustments to simulation

# Parameters
Parameters =   {"Num_mic"   :9, # number of microbes: should agree with input microbe array if running for given microbes
                "r_M"       :1, # intrinsic birth rate of microbes
                "K_M"       :1, # total load of microbes
                "Horiz_T"   :0, # Rate of colonization
                "Env_update":0, # Rate of update of Environment microbe pool by host
                "T"         :1000, # Length of simulation
                "Intraspecific interaction": "Yes" # Enter "Yes" or "No"
                }

# Microbes
# Manually input microbe trait characteristics here - will be used if simulation sun for given microbes
Microbes = np.array([[1,1,1,0,0,0,-1,-1,-1], # effect on own fitness
                    [1,0,-1,1,0,-1,1,0,-1],  # effect on fitness of other microbes
                    [0,0,0,0,0,0,0,0,0]])    # effect on host fitness

# Host
Host = [0,0]   # Host trait

# Number of trials
TT = 10000
#########################

def run_scheme1_singlehost(Parameters, Microbes, Host, init_A):

    # Environmental acquisitions and Host phenotype not considered. Can be changed.

    E_M = Microbes[0]
    E_MM = Microbes[1]
    #E_MH = Microbes[2]
    r_M = Parameters["r_M"]
    K_M = Parameters["K_M"]
    N = Parameters["Num_mic"]

    intrasp = Parameters["Intraspecific interaction"]

    E_HM = Host[0]
    #E_HH = Host[1]

    # Environment pool
    #Env = np.full(N, K_M / N)   # initialize environment pool abundances

    #HT = Parameters["Horiz_T"]
    #Env_upd = Parameters["Env_update"]

    time_steps = Parameters["T"]
    A = init_A

    Output = []     # time series of microbe abundances

    for t in range(time_steps):
        if intrasp == "Yes":
            EV = 1 + (r_M + E_M + E_HM + sum(A * E_MM))
        elif intrasp == "No":
            EV = 1 + (r_M + E_M + E_HM + sum(A * E_MM) - A * E_MM)
        else:
            print("Please specify presence of intraspecific interaction in Parameters")
            break
        A = A * EV #+ HT*Env   # Microbe growth and colonization: Can add colonization
        A = A * K_M / sum(A)  # normalize to get abundances

        #R_H = A * E_MH + E_HH   # host fitness

        #Env = Env + Env_upd * A    # environment pool update
        #Env = Env / sum(Env)

        Output.append(A)

    return (Output)


def stat_dist_given_microbes(Parameters, Microbes, Host, TT):  # Obtain microbe abundances at 1000th time-step for TT many trials with
                                                # random initial abundances

    N = Parameters["Num_mic"]
    stat_dist = pd.DataFrame(columns=list(str(i) for i in range(N)))

    for T in range(TT):
        init_A = np.random.rand(N)              # Generate random initial abundance
        init_A = init_A/sum(init_A)

        output = run_scheme1_singlehost(Parameters, Microbes, Host, init_A)


        stat_dist[T] = output[-1]
        print(T)
    return(stat_dist)

def stat_dist_random_microbes(Parameters, Host,TT):  # Obtain microbe abundances at 1000th time-step for TT many trials with
                                               # random initial abundances

    N = Parameters["Num_mic"]
    stat_dist = pd.DataFrame(columns=list(str(i) for i in range(N)))

    for T in range(TT):
        init_A = np.random.rand(N)  # Level 1 of randomness: Initial abundances
        init_A = init_A / sum(init_A)

        Microbes = np.random.rand(3, N) * 2 - 1  # Level 2 of randomization: Random microbe trait sample | Optional
                                                                             # can read as non-random input instead
        Microbes[0].sort()  # Order effect on self. Replace 0 with 1 to order wrt effect on other microbes
                                                  # Replace 0 with 2 to order wrt effect on host fitness

        output = run_scheme1_singlehost(Parameters, Microbes, Host, init_A)

        stat_dist[T] = output[-1]
        print(T)
    return (stat_dist)

stat = stat_dist_given_microbes(Parameters, Microbes, Host, TT)

sns.set_theme()

fig, axes = plt.subplots(nrows=1,ncols=2)

sns.stripplot(data = stat.T, ax=axes[0])
axes[0].set_xlabel("Microbe")
axes[0].set_ylabel("Microbe abundance at t={}".format(Parameters["T"]))
sns.scatterplot(data=np.mean(stat.T, axis=0), ax=axes[1])
axes[1].set_xlabel("Microbe")
axes[1].set_ylabel("Mean microbe abundance at t={}".format(Parameters["T"]))
fig.tight_layout()

plt.show()