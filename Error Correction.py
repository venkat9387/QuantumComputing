#Author          : Venkatesan Manivasagam
#Student Number  : 2007095
#Date Created    : 05/11/2021
# References : https://qiskit.org/textbook/ch-quantum-hardware/error-correction-repetition-code.html 

pip install qiskit

from qiskit import *
from IPython.display import clear_output
clear_output()

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error

from qiskit.ignis.verification.topological_codes import RepetitionCode
from qiskit.ignis.verification.topological_codes import lookuptable_decoding
from qiskit.ignis.verification.topological_codes import GraphDecoder

def make_noise(p_cx=0,T1T2Tm=(1,1,0)):
    '''
        Returns a noise model specified by the inputs
        - p_cx: probability of depolarizing noise on each
                qubit during a cx
        - T1T2Tm: tuple with (T1,T2,Tm), the T1 and T2 times
              and the measurement time
    '''
    
    noise_model = NoiseModel()

    # single quibit thermal gate relaxation error applied to x gate!
    (T1,T2,Tm) = T1T2Tm
    errors_u1  = thermal_relaxation_error(T1, T1, Tm)
    noise_model.add_all_qubit_quantum_error(errors_u1, ["x"]) 
                 
    return noise_model

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
aer_sim = Aer.get_backend('aer_simulator')

# Function to Derive the raw results.

def get_raw_results(code,noise_model=None):
    circuits = code.get_circuit_list()
    raw_results = {}
    for log in range(2):
        qobj = assemble(circuits[log])
        job = aer_sim.run(qobj, noise_model=noise_model)
        raw_results[str(log)] = job.result().get_counts(str(log))
    return raw_results

raw_results = get_raw_results(code)
for log in raw_results:
    print('Logical', log, ':', raw_results[log], '\n')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')

# Initialization of Monte Carlo Simulation. 

avg = 0.00056 
std_dev = .000051
#std_dev = .051
num_reps = 100
num_simulations = 10000
T1_random = np.random.normal(avg, std_dev, num_reps)

avg = 0.0007
std_dev = .000051
num_reps = 100
num_simulations = 10000
T2_random= np.random.normal(avg, std_dev, num_reps)

T1=[]
T2=[]
T1_el =[]
T2_el =[]
Probabilty=[]
count = -1
count_C =[]



# Custom built function to validate T2=>T1.
for (x,y) in zip( T1_random,T2_random):
  if 2*x >=y:  
    T1.append(x)
    T2.append(y)    
  else:   
    T1_el.append(x)
    T2_el.append(y)


for (x,y) in zip(T1,T2):
  count = count+1
  # Declaration of Repetition Code. 
  code = RepetitionCode(2,4)
 
  # Declaratio of noise models.
  Tm=0.0001
  noise_model2 = make_noise(p_cx=0, T1T2Tm=(1,1,0))
  noise_model1 = make_noise(p_cx=0, T1T2Tm=(x,y,Tm))

  raw_results = get_raw_results(code,noise_model1)
  

  circuits = code.get_circuit_list()

  table_results = {}
  for log in range(2):
    qobj = assemble(circuits[log], shots=10000)
    job = aer_sim.run(qobj, noise_model=noise_model2)
    table_results[str(log)] = job.result().get_counts(str(log))

  #Lookup Table Decoding. 
  P = lookuptable_decoding(table_results,raw_results)
  Probabilty.append(P.get('1'))
  count_C.append(count+1)
  print(raw_results)
 
print(Probabilty)
plt.plot(count_C,Probabilty)
plt.ylabel('Probabilty Of "1"')
plt.xlabel('Count')
plt.show()

