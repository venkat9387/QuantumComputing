#Author          : Venkatesan Manivasagam
#Student Number  : 2007095
#Date Created    : 05/11/2021
# Reference 1 : https://qutip.org/docs/latest/guide/guide-control.html
# Reference 2 : https://qutip.org/docs/latest/guide/dynamics/dynamics-master.html

pip install qutip

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

import math as math



import datetime

from qutip import Qobj, identity, sigmax, sigmaz , fock ,num, qeye, destroy , tensor , basis , mesolve , mcsolve , sigmay,rand_ket , sigmam ,Bloch , expect,Options
from qutip.qip import hadamard_transform
from qutip.qip import x_gate
import qutip.logging_utils as logging
logger = logging.get_logger()
#Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
#QuTiP control modules
import qutip.control.pulseoptim as cpo

example_name = 'Hadamard'

# Drift Hamiltonian
H_d = sigmaz()
# The (single) control Hamiltonian
H_c = [sigmax()]
# start point for the gate evolution
U_0 = identity(2)
# Target for the gate evolution Hadamard gate
#U_targ = hadamard_transform(1)
U_targ = x_gate(1)

# Number of time slots
n_ts = 100 # We have increase this. Very important for deriving the answers
# Time allowed for the evolution
evo_time = 500  # Change that to 1000 and observe!

#evolution time divided by the n_ts would be my time interval. 

# Fidelity error target
fid_err_targ = 1e-5 # Dont go above e-6 and stay above e-3
# Maximum iterations for the optisation algorithm
max_iter = 200
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20

# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'RND'

#Set to None to suppress output files
f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)



result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, n_ts, evo_time, 
                fid_err_targ=fid_err_targ, min_grad=min_grad, 
                max_iter=max_iter, max_wall_time=max_wall_time, 
                out_file_ext=f_ext, init_pulse_type=p_type, # use random for p_type/we used.
                log_level=log_level, gen_stats=True)

result.stats.report()
#print("Final evolution\n{}\n".format(result.evo_full_final))
#print("********* Summary *****************")
print("Final fidelity error {}".format(result.fid_err))

print("Number of iterations {}".format(result.num_iter))


print(result.evo_full_initial)
amplitudes=result.final_amps[:, 0] # Yeah, 100 amplitude is okay to have more freedom on the accuracy.!!1







import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid') # Probability Graph Style set.

avg = 560000 #  Setting up Relaxation Noise ; Time scale set to Nano Seconds.
std_dev = 5000 # Standard Deviation of Relaxation Noise ; Time scale set to Nano Seconds.
num_reps = 100
num_simulations = 10000
T1_random = np.random.normal(avg, std_dev, num_reps)

avg = 700000  #  Setting up Relaxation Noise ; Time scale set to Nano Seconds.
std_dev = 5000 # Standard Deviation of Relaxation Noise ; Time scale set to Nano Seconds.
num_reps = 100
num_simulations = 10000
T2_random= np.random.normal(avg, std_dev, num_reps) 

T1=[]
T2=[]
T1_el =[]
T2_el =[]
Fidelity=[]

for (x,y) in zip( T1_random,T2_random): # faster numpy options.
  if 2*x >=y: # Condition to verify if the T1 and T2 Relaxation concurs with the formula T2<= 2T1 
    T1.append(x)
    T2.append(y)    
  else:    
    T1_el.append(x)
    T2_el.append(y)

for (t1,t2) in zip(T1,T2):

  #Initilaizing State Vector.
  psi0 = basis(2,0) # Initial State Vector #2 : basis(2,1)
  t= np.linspace(0,evo_time,n_ts) #evo = 10
  
  # Deriving Collapse Operators.
  c_op_list = []
  a= destroy(2)  
  C1 = a / np.sqrt(t1) 
  C2eff = 1/( (1/t2) - 1/(2*t1) )
  C2 = a.dag() * a * np.sqrt(2/C2eff)

  # Verfying if the Relaxations are not zero.
  if t1 > 0.0:
    # Applied Annihilation Operator to Relaxation Noise t1. 
    c_op_list.append(C1 * sigmam()) 
  if t2 > 0.0:
    # Applied Annihilation Operator to the Relaxation Noise t2. 
    c_op_list.append(C2 * sigmaz())
 

  # Initilaizing Drift Hamiltonian and Control Hamiltonian. 
  H0 = sigmax()
  H1 = sigmay() 

  # Deriving Coefficients of Control Hamiltonian. 
  args1={'ams': amplitudes, # Amplitudes dervied from Grape Control Algorithm.         
        'N': n_ts, 
        'T': evo_time }

  def H1_coeff(t,args):   
    return args1['ams'][min(int(np.floor( t / (args1['T'] / args1['N']))),len(args1['ams'])-1 )]
  

  # Derving System Hamiltonian from Drif and Control Hamiltonians. 
  H = [H0,[H1,H1_coeff]]

  # Assigning options for Dynamic Solvers. 
  mcsolve_options = Options(store_states=True,store_final_state=True,average_states=True,average_expect=True)

  # Assigning System Hamiltonian , Evolution time slots, Collapse operators, Projections of Control Hamiltonian and Dynamic Solver options to Monte Carlo Solver.     
  output = mcsolve(H, psi0, t, c_op_list, [sigmax(), sigmay(), sigmaz()],options=mcsolve_options)
  
  # Extration of gate Fidelity.
  FinalState = output.final_state.data
  FinalState_array = FinalState.data
  f = np.real(FinalState_array[3])
  #f = np.real(FinalState_array[0])
  Fidelity.append(f)

print(Fidelity)
