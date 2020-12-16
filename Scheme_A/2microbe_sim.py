import general_functions as gf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Parameters = {'Num_mic'     :2,     # number of microbes
              'Num_host'    :500,   # number of hosts
              'death_rate'  :0.001, # host event rate
              'host_weight' :0.5,     # host weightage in calculating fitness (1 - microbe weightage)
              'mutation'    :0.01, # variation rate in host reproduction
              'mic_intR'    :1,     # microbe intrinsic fitness
              'mic_capacity':1000,   # microbe carrying capacity
              'col_rate'    :0.1,   # colonization rate
              'Env_update'  :1,   # host influence on environment microbe abundance
              'sim_time'    :1000,  # number of time steps
              'num_bins'    :20     # number of bins to generate histogram of distributions
              }

K = Parameters['mic_capacity']
bins = Parameters['num_bins']


# Define interaction matrix for the 2 microbe system
I = np.array([[0,0,0],[0,-1,1]])


AB,T,RH = gf.run_schemeA_givenI(I, Parameters)

ABA = pd.DataFrame(np.array(AB[0]), columns=list(str(i) for i in np.flip(np.linspace(0,K,bins-1)).round(0)))
ABB = pd.DataFrame(np.array(AB[1]), columns=list(str(i) for i in np.flip(np.linspace(0,K,bins-1)).round(0)))
TA = pd.DataFrame(np.array(T[0]), columns=list(str(i) for i in np.flip(np.linspace(0, 1, bins - 1)).round(1)))
TB = pd.DataFrame(np.array(T[1]), columns=list(str(i) for i in np.flip(np.linspace(0, 1, bins - 1)).round(1)))
RHA = pd.DataFrame(np.array(RH[0]), columns=list(str(i) for i in np.flip(np.linspace(-1, 1, bins - 1)).round(1)))
RHB = pd.DataFrame(np.array(RH[1]), columns=list(str(i) for i in np.flip(np.linspace(-1, 1, bins - 1)).round(1)))

plt.subplot(3,2,1)
plt.imshow(ABA.T, cmap="Spectral", aspect='auto')
plt.yticks(np.arange(0,bins,bins/4), np.flip(np.arange(K/4,K*(1.25),K/4)))
plt.xlabel("Time")
plt.ylabel("Microbe A \n abundance per host")
plt.colorbar()


plt.subplot(3,2,2)
plt.imshow(ABB.T, cmap="Spectral", aspect='auto')
plt.yticks(np.arange(0,bins,bins/4), np.flip(np.arange(K/4,K*(1.25),K/4)))
plt.xlabel("Time")
plt.ylabel("Microbe B \n abundance per host")
plt.colorbar()

plt.subplot(3,2,3)
plt.imshow(TA.T, cmap="Spectral", aspect='auto')
plt.yticks(np.arange(0,bins,bins/4), np.flip(np.arange(0.25,1.25,.25)))
plt.xlabel("Time")
plt.ylabel("Microbe A transmission")
plt.colorbar()

plt.subplot(3,2,4)
plt.imshow(TB.T, cmap="Spectral", aspect='auto')
plt.yticks(np.arange(0,bins,bins/4), np.flip(np.arange(0.25,1.25,.25)))
plt.xlabel("Time")
plt.ylabel("Microbe B transmission")
plt.colorbar()

plt.subplot(3,2,5)
plt.imshow(RHA.T, cmap="Spectral", aspect='auto')
plt.yticks(np.arange(0,bins,bins/4), np.flip(np.arange(-0.5,1.5,.5)))
plt.xlabel("Time")
plt.ylabel("Host control on Microbe A")
plt.colorbar()

plt.subplot(3,2,6)
plt.imshow(RHB.T, cmap="Spectral", aspect='auto')
plt.yticks(np.arange(0,bins,bins/4), np.flip(np.arange(-0.5,1.5,.5)))
plt.xlabel("Time")
plt.ylabel("Host control on Microbe B")
plt.colorbar()

plt.tight_layout()
plt.show()