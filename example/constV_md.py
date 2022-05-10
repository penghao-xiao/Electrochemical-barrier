#!/usr/bin/env python
'''
 MD with a discrete voltage control
'''

from ase.optimize.fire import FIRE
from ase import *
from ase.io import read,write
import os
import sys
import numpy as np
from ecb.eAtoms import eAtoms
from ase.calculators.vasp import Vasp

ncc = 100 # md steps with constant charge. Number of electrons is optimized (voltage control) every ncc steps.
total_steps = 10000 # total MD steps. 
nloops = int(total_steps/ncc) # number of outmost loops 
maxne = 50 # max number of steps for optimizing number of electrons 

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
            algo = 'Fast',
            lreal= 'Auto',
            #lreal= False,
            lplane = True,
            encut= 400,
            ismear = 0,
            sigma  = 0.05,
            lmaxmix   = 4,
            lvtot = True, 
            lvhar = True, 
            ispin = 2,
            nelm  = 60,
            lsol= True,
            eb_k=80,
            tau=0,
            lambda_d_k=3.0,
            nelect = 317.2, #initial number of electrons 
            # md:
            ibrion = 0, 
            potim  = 0.5, 
            nsw    = ncc, 
            mdalgo = 1, 
            andersen_prob = 0.1, 
            tebeg  = 300, 
              )
p1.set_calculator(calc)
#print p1.get_potential_energy()

p1box = eAtoms(p1, voltage=-1.0, e_only=True)

# MD with a discrete voltage control
for i in range(nloops):
    # constant charge MD by vasp
    p1box._calc.set(nsw = ncc)
    p1box.get_potential_energy() # enthalpy including the work term
    p1box._calc.set(nsw = 0)
    # ASE optimizor for ne, for ase.3.15 and later
    dyn = FIRE(p1box, maxmove = 0.1, dt = 0.1, dtmax = 0.1, force_consistent = False)
    dyn.run(fmax=0.01, steps=maxne)

write("CONTCAR_Vot.vasp",p1,format='vasp',vasp5=True, direct=True)

