import pickle
import numpy as np
from tsase.neb.util import vmag
from ase.io import read, write

#with open(r"df_unit.pickle", "rb") as output_file:
#    df_unit = pickle.load(output_file)
with open(r"curvature_along_force+0.5e/Knr.pickle", "rb") as input_file:
    Knr_all = pickle.load(input_file)

# substract the f0 before adding the charge
with open(r"f0.pickle", "rb") as input_file:
    f0 = pickle.load(input_file)
dn = 0.5
print vmag(Knr_all)
Knr_all -= f0/dn
print vmag(Knr_all)

print("Knr_all shape:", Knr_all.shape)
with open('dynamicMatrix/DISPLACECAR', 'r') as dymfile:
   freedom = np.loadtxt(dymfile,skiprows=0)
print(freedom[1:3])
# only keep the freedoms that are nonzero in the DISPLACECAR
Knr = []
for i in range(len(Knr_all)):
   if freedom[i][0] >0: Knr.append(Knr_all[i])
Knr = np.array(Knr).flatten()
print("Knr shape", Knr.shape)
#print Knr
print vmag(Knr)

with open('dynamicMatrix/freq.mat', 'r') as dymfile:
   Krr = np.loadtxt(dymfile,skiprows=0)
print("Krr shape", Krr.shape)

# calculate Knr*invKrr*Krn
#invKrr = np.linalg.inv(Krr)
#tmp = np.matmul(Knr, invKrr)
#Keff = np.matmul(tmp, Knr.transpose())
#tmp = np.linalg.solve(Krr, Knr.transpose())
tmp = np.linalg.lstsq(Krr, Knr.transpose(), rcond=0.0001)[0]
Keff = np.matmul(Knr, tmp)
print("Keff", Keff)

#C_saddle  = 0.6182; 
#U0_saddle = 1.1667
#U0_saddle = 1.1839
C_saddle  = 0.6311; 
U0_saddle = 0.6825
C = C_saddle
Ceff = 1/(1/C - Keff)
print("Ceff", Ceff)

U0 = U0_saddle
U  = 1.6
dn = Ceff*(U-U0)
dr = np.linalg.lstsq(Krr, Knr.transpose()*-dn, rcond=0.0001)[0]
print("U=", U)
print("dn=", dn)
p0 = read('dynamicMatrix/POSCAR', format='vasp')
r0 = p0.get_positions()
j = 0
for i in range(len(r0)):
   if freedom[i][0] >0: 
       r0[i] += dr[j:j+3]
       j += 3

p0.set_positions(r0)
write('POSCAR_'+str(U)+'V.vasp', p0, format='vasp', vasp5=True, direct=True)






