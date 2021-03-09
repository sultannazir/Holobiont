import general_functions as gf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

Parameters = {'Num_mic'     :2,     # number of microbes
              'Num_host'    :500,   # number of hosts
              'death_rate'  :1, # probability of one host event happening at each time step
              'host_weight' :0.5,     # host weightage in calculating fitness (1 - microbe weightage)
              'mutation'    :0.1, # variation rate in host reproduction
              'seed'        :1,     # number of vertically transmitted microbes colonizing offspring
              'col_rate'    :0.1,   # horizontal colonization rate
              'mic_intR'    :1,     # microbe intrinsic fitness
              'mic_capacity':1000,   # microbe carrying capacity
              'Env_update'  :0.999,   # host influence on environment microbe abundance
              'sim_time'    :10000,  # number of time steps
              'HMnum_bins'  :10,     # number of bins for heatmap axes
              'num_bins'    :20     # number of bins to generate histogram of distributions
              }

N = Parameters['Num_mic']
M = Parameters['Num_host']
step = Parameters['HMnum_bins']
HM = np.zeros((step,step))
TN = np.zeros((step,step))
TA = np.zeros((step,step))

# set parameters
cost = (-2, 0) # cost log range
seed = (0, 2) # colonization rate log range

actcost = np.logspace(*cost,step)
actcost = np.round(np.flip(actcost),3 )
actseed = np.round(np.logspace(*seed, step),0)
print(actseed, actcost)

# Initialize
A = np.random.choice([0,1],(M,N))
RH = np.zeros((M,N+1))
T = np.ones((M,N))

Env = np.full(N,1/N)
x = []
y = []
z=[]
for i in range(step):
    for j in range(step):
        I = np.array([[0, 0, 0], [0, -actcost[j], 1]])
        Parameters['seed'] = actseed[i]

        AB, Trans = gf.run_schemeA_giveninit_getmean(I, RH, T, A, Env, Parameters)
        HM[i][j] = AB[1][-1]
        TN[i][j] = Trans[0][-1]
        TA[i][j] = Trans[1][-1]
        x.append(np.log10(actcost)[j])
        y.append(np.log10(actseed)[i])
        z.append(AB[1][-1])             # get altruist mean frequency at the last time step
        print(i,j)


HM = pd.DataFrame(HM, columns=list(str(round(i,5)) for i in actcost))
TN = pd.DataFrame(TN, columns=list(str(round(i,5)) for i in actcost))
TA = pd.DataFrame(TA, columns=list(str(round(i,5)) for i in actcost))
HM['Vertical transmission size'] = list(str(round(i,4)) for i in actseed)
TN['Vertical transmission size'] = list(str(round(i,4)) for i in actseed)
TA['Vertical transmission size'] = list(str(round(i,4)) for i in actseed)
HM = HM.set_index('Vertical transmission size')
TN = TN.set_index('Vertical transmission size')
TA = TA.set_index('Vertical transmission size')

sns.heatmap(HM, cmap='rainbow', cbar_kws={'label': 'Mean helper frequency at t = {}'.format(Parameters['sim_time'])}, vmin=0, vmax=1)
plt.xlabel("Magnitude of cost to helper")


plt.title('Effects of magnitudes of Vertical transmission and Microbe selection \n on stationary distribution of helper microbes '
          '\n No host control, T = 1 for all microbes')
plt.show()

fig = plt.figure()
ax = plt.axes(projection="3d")
im = ax.scatter3D(x,y,z, c=z, cmap='viridis')
ax.set_xlabel('Cost to altruist\n (in log10 scale)')
ax.set_ylabel('Vertical transmission size\n (in log10 scale)')
fig.colorbar(im)
plt.show()