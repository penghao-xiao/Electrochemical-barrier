#!/usr/bin/env python
'''

Cell optimization example using mushy box with shear stress applied
'''

from ase.optimize.fire import FIRE
from ase import *
from ase.io import read,write
import os
import sys
import numpy as np
#from eAtoms import eAtoms
from ase.calculators.vasp import Vasp
from util import vunit, vdot, vmag
import pickle

p1 = read('final.vasp',format='vasp')
n0 = 436
dn = 0.5

calc = Vasp(prec = 'Normal', 
            ediff = 1e-5,
            kpts = (4,4,1),
            gamma= True,
            xc = 'PBE',
            lcharg = False,
            isym = 0,
            npar = 4,
            nsim = 4,
            algo = 'Fast',
            lreal= 'Auto',
            lplane = True,
            encut= 400,
            ismear = 0,
            sigma  = 0.05,
            lvtot = True, 
            lvhar = True, 
            ispin = 1,
            lsol=True,
            eb_k=80,
            tau=0,
            lambda_d_k=3.0,
            nelect = n0+dn
              )
p1.set_calculator(calc)
p2 = p1.copy()
#print p1.get_potential_energy()
f1 = p1.get_forces()
'''
f1direction = vunit(f1)
#displacement = 0.1
displacement = 0.5
rt  = p1.get_positions()
rt += displacement*f1direction
with open(r"df_unit.pickle", "wb") as output_file:
    pickle.dump(f1direction, output_file)
p2.set_positions(rt)
calc.set(nelect=n0)
p2.set_calculator(calc)
f2 = p2.get_forces()
#Krr_f1 = vdot(f2, f1direction)/displacement
#Keff = vdot(Knr, Knr)/Krr_f1
#print("Krr_f1:", Krr_f1)
#print("Keff:", Keff)
'''

Knr = f1/dn
print("Knr:", vmag(Knr))
with open(r"Knr.pickle", "wb") as output_file:
    pickle.dump(Knr, output_file)

#p1e = eAtoms(p1, epotential = 0.0, ne=436.2, n0=436)

# for ase.3.15 and later
#dyn = FIRE(p1,maxmove = 0.1, dt = 0.1, dtmax = 0.1, force_consistent = False)
#dyn = MDMin(p1box, dt=0.1, force_consistent = False)
# for ase.3.12 and earlier
#dyn = FIRE(p1box,maxmove = 0.1, dt = 0.1, dtmax = 0.1)
#dyn = MDMin(p1box, dt=0.1)
#dyn.run(fmax=0.05, steps=310)

#write("CONTCAR_Vot.vasp",p1,format='vasp',vasp5=True, direct=True)

