#!/usr/bin/env python

'''
Neb optimization example
'''

from ecb.eneb import eneb, qm_essneb
#from tsase.calculators.vasp_ext import Vasp
from ase.calculators.vasp import Vasp
from ase.io import read
import os

p1 = read('0.CON',format='vasp')
p2 = read('6.CON',format='vasp')

#calc = LAMMPS(parameters=parameters, tmp_dir="trash")
calc = Vasp(prec = 'Normal', 
            ediff = 1e-5,
            #kpts = (2,2,1),
            kpts = (1,1,1),
            gamma = True,
            xc = 'rPBE',
            lcharg = False,
            isym = 0,
            npar = 6,
            nsim = 4,
            lreal= 'Auto',
            algo= 'Normal',
            encut= 400,
            lplane=True,
            isif = 0,
            ispin = 2, 
            nelm = 60,
            lvtot = True, 
            lvhar = True, 
            lsol= True,
            eb_k=80,
            tau=0,
            lambda_d_k=3.0
              )
p1.set_calculator(calc)
p2.set_calculator(calc)

####################################################
calc.initialize(p1)
calc.write_potcar()
nelect0 = calc.default_nelect_from_ppp()
####################################################

nim = 7
band = eneb(p1, p2, numImages = nim, ne1=317.6758, ne2=318.2161, ne0=nelect0, voltage=-1.0, method='ci')
# read images from the last run
'''
for i in range(1,nim-1):
    filename = str(i)+'.CON'
    b = read(filename,format='vasp')
    band.path[i].positions=b.get_positions()
    #band.path[i].set_positions(b.get_positions())
    #band.path[i].set_cell(b.get_cell())
'''

opt = qm_essneb(band, maxmove = 0.1, dt = 0.1)
#opt = fire_essneb(band, maxmove =0.1, dtmax = 0.1, dt=0.1)
opt.minimize(forceConverged=0.01, maxIterations = 300)
os.system('touch movie.con')
for p in range(len(band.path)):
    os.system('cat '+str(p)+'.CON >>movie.con')



