import general_functions as gf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Parameters = dict(Num_mic=2, Num_host=500, death_rate=1, host_weight=0, mutation=0, seed=1, mic_intR=1,
                  mic_capacity=1000, col_rate=0.1, Env_update=0, sim_time=500000, num_bins=20)

K = Parameters['mic_capacity']
bins = Parameters['num_bins']
N = 2
M = Parameters['Num_host']

# Initialize
I = np.array([[0,0,0],[0,-1,1]])
A = np.ones((M,N))
RH = np.zeros((M,N+1))
T = np.ones((M,N))
Env = np.full(N,1/N)

AB,T,RH = gf.run_schemeA_giveninit(I, RH, T, A, Env, Parameters)

ABA = pd.DataFrame(np.array(AB[0]), columns=list(str(i) for i in np.flip(np.linspace(0,1,bins-1)).round(1)))
ABB = pd.DataFrame(np.array(AB[1]), columns=list(str(i) for i in np.flip(np.linspace(0,1,bins-1)).round(1)))
TA = pd.DataFrame(np.array(T[0]), columns=list(str(i) for i in np.flip(np.linspace(0, 1, bins - 1)).round(1)))
TB = pd.DataFrame(np.array(T[1]), columns=list(str(i) for i in np.flip(np.linspace(0, 1, bins - 1)).round(1)))
RHA = pd.DataFrame(np.array(RH[0]), columns=list(str(i) for i in np.flip(np.linspace(-1, 1, bins - 1)).round(1)))
RHB = pd.DataFrame(np.array(RH[1]), columns=list(str(i) for i in np.flip(np.linspace(-1, 1, bins - 1)).round(1)))

plt.subplot(3,2,1)
plt.imshow(ABA.T, cmap="Spectral", aspect='auto')
plt.yticks(np.arange(0,bins,bins/4), np.flip(np.arange(0.25,1.25,0.25)))
plt.xlabel("Time")
plt.ylabel("Neutral microbe \n abundance per host")
plt.colorbar()

plt.subplot(3,2,2)
plt.imshow(ABB.T, cmap="Spectral", aspect='auto')
plt.yticks(np.arange(0,bins,bins/4), np.flip(np.arange(0.25,1.25,0.25)))
plt.xlabel("Time")
plt.ylabel("Altruist microbe \n abundance per host")
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