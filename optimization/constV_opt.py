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
from eAtoms import eAtoms
from ase.calculators.vasp import Vasp

p1 = read('0.CON',format='vasp')

calc = Vasp(prec = 'Normal', 
            ediff = 1e-5,
            #kpts = (2,2,1),
            kpts = (1,1,1),
            gamma= True,
            xc = 'rPBE',
            #lvdw = True,
            lcharg = False,
            isym = 0,
            npar = 4,
            nsim = 4,
            algo = 'All',
            lreal= 'Auto',
            #lreal= False,
            lplane = True,
            encut= 400,
            #ismear = 0,
            #sigma  = 0.05,
            lmaxmix   = 4,
            lvtot = True, 
            lvhar = True, 
            ispin = 2,
            nelm  = 60,
            lsol= True,
            eb_k=80,
            tau=0,
            lambda_d_k=3.0, 
            nelect = 318.0839 # set the initial guess for the total charge here
              )
p1.set_calculator(calc)
#print p1.get_potential_energy()

####################################################
calc.initialize(p1)
calc.write_potcar()
nel = calc.read_default_number_of_electrons()
nel_d = {}
for el in nel:
    nel_d[el[0]] = el[1]
nelect0 = int(sum([nel_d[atom.symbol] for atom in p1]))
####################################################

p1box = eAtoms(p1, epotential = -1.0, n0=nelect0)

# for ase.3.15 and later
dyn = FIRE(p1box,maxmove = 0.1, dt = 0.1, dtmax = 0.1, force_consistent = False)
#dyn = MDMin(p1box, dt=0.1, force_consistent = False)
# for ase.3.12 and earlier
#dyn = FIRE(p1box,maxmove = 0.1, dt = 0.1, dtmax = 0.1)
#dyn = MDMin(p1box, dt=0.1)
dyn.run(fmax=0.01, steps=310)

write("CONTCAR_Vot.vasp",p1,format='vasp',vasp5=True, direct=True)

