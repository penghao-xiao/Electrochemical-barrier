#!/usr/bin/env python
'''

Cell optimization example using mushy box with shear stress applied
'''

from ase.optimize.fire import FIRE
from ase.optimize.mdmin import MDMin
from ase import *
from ase.io import read,write
import os
import sys
import numpy as np
from eAtoms import eAtoms
from ase.calculators.vasp import Vasp

calc = Vasp(prec = 'Normal', 
            ediff = 1e-5,
            kpts = (3,3,1),
            gamma= True,
            xc = 'rPBE',
            #lvdw = True,
            lcharg = False,
            isym = 0,
            npar = 4,
            nsim = 4,
            algo = 'Normal',
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
              )
# nelect sets the initial guess for the number of electrons
nelect = 318.0839 
calc.set(nelect = nelect)
# read the initial structure
p1 = read('initial.vasp',format='vasp') 
p1.set_calculator(calc)

# epotential sets the voltage vs. SHE
p1box = eAtoms(p1, voltage = -1.0) 

# for ase.3.15 and later
dyn = FIRE(p1box,maxmove = 0.1, dt = 0.1, dtmax = 0.1, force_consistent = False)
# for ase.3.12 and earlier
#dyn = FIRE(p1box,maxmove = 0.1, dt = 0.1, dtmax = 0.1)
#dyn = MDMin(p1box, dt=0.1)
dyn.run(fmax=0.01, steps=220)

write("CONTCAR_Vot.vasp",p1,format='vasp',vasp5=True, direct=True)

