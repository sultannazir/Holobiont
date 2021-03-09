import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import general_functions as gf

Parameters = {'Num_mic'     :2,     # number of microbes
              'host_weight' :0,     # host weightage in calculating fitness (1 - microbe weightage)
              'mic_intR'    :1,     # microbe intrinsic fitness
              'mic_capacity':1000,   # microbe carrying capacity
              'col_rate'    :0.01,   # colonization rate
              'sim_time'    :500,  # number of time steps
              'num_sim'     :100    # number of simulations
              }

N = Parameters['Num_mic']
num = Parameters['num_sim']

# intialize
A = np.array([[0,0]])
I = np.array([[0, 0, 0],
              [0, 0, 1]])
RH = np.array([[0, 0, 0]])
Env = np.array([0.5,0.5])


# colors for N microbes
color=plt.cm.rainbow(np.linspace(0,1,N))

# different initial abundances to run simulation over
init_ab = [[0,0],[1,0],[0,1],[1,1]]

# subplot spacing
fig = plt.figure(figsize = (9,9))
gs1 = gridspec.GridSpec(1, 4)
gs1.update(wspace=0.025, hspace=0.05)


for n in range(4): # plot row
    for m in range(1):  # plot column
        print(m,n)
        A = np.array([init_ab[n]])*10**m              # initial abundance for plots in nth row
        Parameters['col_rate'] = 0.1     # colonization rate for plots in mth column
        AB = []
        for i in range(num):                    # rum simulation num times
            AB.append(gf.run_schemeA_frac_singlehost(I, RH, A, Env, Parameters))
            AB[i] = pd.DataFrame(AB[i], columns=list(str(i + 1) for i in range(N)))

        plt.subplot(2,4,4*m+n+1)
        for i in range(num):
            for j in range(N):
                plt.plot(AB[i][str(j+1)],color=color[j], alpha=5/num)
        plt.title("Initial abundance {} \n Colonization rate {}".format(np.array(init_ab[n])*10**m, Parameters['col_rate']))

# microbe colours in legend
custom_lines = [Line2D([0], [0], color=color[i], lw=2) for i in range(N)]

#plt.suptitle("{} simulations of 2 microbe system with a neutral and an altruist microbe"
#             "\n for different colonization rates and initial abundances (in the form [neutral, altruist]) "
#             "\n Cost to altruist is {}".format(num,I[1][1]))
plt.legend(custom_lines, ['Neutral', 'Altruist'], loc=(1.04,0))

fig.text(0.5, 0.005, 'Time', ha='center')
fig.text(0.005, 0.5, 'Relative abundance', va='center', rotation='vertical')
plt.tight_layout()
plt.show()