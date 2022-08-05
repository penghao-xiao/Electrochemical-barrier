#!/usr/bin/env python
'''
 MD with a discrete voltage control
'''

from ase.optimize.fire import FIRE
from ase.io import read,write
from ecb.eAtoms import eAtoms
from ase.calculators.vasp import Vasp
import os

steps_cc = 2 # md steps with constant charge. Number of electrons is optimized (voltage control) every ncc steps.
steps_total = 6 # total MD steps. 
nloops = int(steps_total/steps_cc) # number of outmost loops 
maxne = 1 # max number of steps for optimizing number of electrons 

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
            ncore = 16,
            nsim = 4,
            algo = 'Fast',
            lreal= 'Auto',
            #lreal= False,
            lplane = True,
            encut= 400,
            ismear = 0,
            sigma  = 0.05,
            #lmaxmix   = 4,
            lvhar = True, 
            ispin = 1,
            nelm  = 60,
            lsol= True,
            eb_k=80,
            tau=0,
            lambda_d_k=3.0,
            nelect = 317.2, #initial number of electrons 
            # md:
            ibrion = 0, 
            potim  = 0.5, 
            nsw    = steps_cc, 
            mdalgo = 1, 
            andersen_prob = 0.1, 
            tebeg  = 300, 
              )
p1.set_calculator(calc)
p1.get_potential_energy()

# temporary folder for optimize ne
tmpdir = 'tmp'
if not os.path.exists(tmpdir):
    os.mkdir(tmpdir)
# MD with a discrete voltage control
for i in range(nloops):
    # ASE optimizor for ne, for ase.3.15 and later
    os.chdir(tmpdir)
    os.system('cp ../WAVECAR ./')
    p1ext = eAtoms(p1, voltage=-1.0, e_only=True)
    p1ext._calc.set(nsw = 0)
    dyn = FIRE(p1ext, maxstep = 0.1, dt = 0.1, dtmax = 0.1, force_consistent = False)
    dyn.run(fmax=0.01, steps=maxne)

    p1._calc.set(nelect = p1ext.ne[0][0])
    p1._calc.set(nsw = steps_cc)
    os.chdir('../')
    os.system('cp CONTCAR POSCAR')

    trajdir = 'traj_'+str(i)
    if not os.path.exists(trajdir):
        os.mkdir(trajdir)
        os.system('cp XDATCAR OUTCAR INCAR '+trajdir)
    # constant charge MD by vasp
    p1._calc.write_incar(p1)
    # Execute VASP
    command = p1._calc.make_command(p1._calc.command)
    with p1._calc._txt_outstream() as out:
        errorcode = p1._calc._run(command=command,
                                  out=out,
                                  directory=p1._calc.directory)
    # Read output
    atoms_sorted = read('CONTCAR', format='vasp')
    # Update atomic positions and unit cell with the ones read from CONTCAR.
    p1.positions = atoms_sorted[p1._calc.resort].positions

write("CONTCAR_Vot.vasp",p1,format='vasp',vasp5=True, direct=True)

