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


p1 = read('POSCAR.vasp',format='vasp')

calc = Vasp(prec = 'Normal', 
            ediff = 1e-5,
            kpts = (3,4,1),
            gamma= False,
            xc = 'RPBE',
            #lvdw = True,
            lcharg = False,
            isym = 0,
            #npar = 4,
            kpar = 8,
            npar = 36,
            nsim = 4,
            algo = 'Normal',
            lreal= 'Auto',
            #lreal= False,
            lplane = True,
            encut= 400,
            lmaxmix   = 4,
            lvtot = True, 
            lvhar = True, 
            ispin = 1,
            nelm  = 60,
            lsol= True,
            eb_k=80,
            tau=0,
            lambda_d_k=3.0,
              )
p1.set_calculator(calc)

# RESTART IF NEEDED
if os.path.isfile('OUTCAR'):
    try:
        calc0 = Vasp(restart=1)
        nelect0 = calc0.get_number_of_electrons()
    except UnboundLocalError as error:
        strmatch = 'grep NELECT OUTCAR' 
        with os.popen(strmatch,'r') as p:
            raw = p.readlines()[-1]
            nelect0 = float(raw.split()[2])
    except IOError as error:
        strmatch = 'grep NELECT OUTCAR' 
        with os.popen(strmatch,'r') as p:
            raw = p.readlines()[-1]
            nelect0 = float(raw.split()[2])


    calc.set(nelect=nelect0)
    print("Restarting with NELECT=%s" %(nelect0))

# GET DEFAULT NUMBER OF ELECTRONS
####################################################
calc.initialize(p1)
calc.write_potcar()
nel = calc.read_default_number_of_electrons()
nel_d = {}
for el in nel:
    nel_d[el[0]] = el[1]
nelect = int(sum([nel_d[atom.symbol] for atom in p1]))
####################################################
p1box = eAtoms(p1, epotential = -1.0, n0=nelect)

# for ase.3.15 and later
dyn = FIRE(p1box,maxmove = 0.1, dt = 0.1, dtmax = 0.1, force_consistent = False)
#dyn = MDMin(p1box, dt=0.1, force_consistent = False)
# for ase.3.12 and earlier
#dyn = FIRE(p1box,maxmove = 0.1, dt = 0.1, dtmax = 0.1)
#dyn = MDMin(p1box, dt=0.1)
dyn.run(fmax=0.01, steps=200)

write("CONTCAR_Vot.vasp",p1,format='vasp',vasp5=True, direct=True)
write("POSCAR.vasp",p1,format='vasp',vasp5=True, direct=True)
