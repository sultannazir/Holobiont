import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import general_functions as gf

Parameters = {'Num_mic'     :2,     # number of microbes
              'host_weight' :0,     # host weightage in calculating fitness (1 - microbe weightage)
              'mic_intR'    :1,     # microbe intrinsic fitness
              'mic_capacity':1000,   # microbe carrying capacity
              'col_rate'    :0.1,   # colonization rate
              'sim_time'    :10000,  # number of time steps
              }

N = Parameters['Num_mic']

# intialize
A = np.array([[500,500]])
I = np.array([[0, 0, 0],
              [0, -1, 1]])
RH = np.array([[0, 0, 0]])
Env = np.array([0.5,0.5])

AB =  gf.run_schemeA_singlehost(I, RH, A, Env, Parameters)

AB = pd.DataFrame(AB, columns=list(str(i+1) for i in range(N)))

AB.plot()
plt.title("Within host dynamics")
plt.ylabel('Abundance')
plt.xlabel('Time')
plt.show()