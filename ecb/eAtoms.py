#!/usr/bin/env python
'''
Generalized Atoms calss for optimization of geometry and number of electrons simultaneously under constant voltage
Contact: Penghao Xiao (pxiao@dal.ca, pxiao@utexas.edu)
Version: 1.0
Usage: first set the vacuum/solution along z axis and move the slab to the center of the simulation box
Please cite the following reference:
'''

from ase import *
from ase.io import read,write
from ase import units
from .read_LOCPOT import align_vacuum
import numpy as np

class eAtoms(Atoms):
    def __init__(self, atomsx, voltage=0.0, solPoisson=True, weight=1.0, e_only=False):
        """ relaxation under constant electrochemical potential
            epotential ... real,
                           electrochemical potential: the work function of the counter electrode under the given voltage
                           i.e. voltage vs. SHE + workfunction of SHE
            solPoisson ... bool,
                           True : compensate charge in the solvent, where VASPsol is required with lambda_d_k=3.0; 
                           False: uniform background charge;
            weight     ... real >0, 
                           weight of the number of electrons vs. atomic positions, 
            e_only     ... bool, 
                           True: only optimize the number of electrons, corresponding to weight=infinity. 
        """
        self.atomsx = atomsx 

        self.epotential= -voltage - 4.6
        self.natom = atomsx.get_number_of_atoms()
        self.ne       = np.zeros((1,3)) # number of electrons
        self.mue      = np.zeros((1,3)) # electron mu, dE/dne
        self.vtot  = 0.0 # shift of the electrostatic potential due to the compensating charge in DFT
        self.direction = 'z'
        self.solPoisson = solPoisson
        self.weight = weight
        self.e_only = e_only
        self.jacobian = self.weight

        Atoms.__init__(self,atomsx)

    def get_positions(self):
        r    = self.atomsx.get_positions()
        Rc   = np.vstack((r, self.ne * self.jacobian))
        return Rc

    def set_positions(self,newr):
        ratom   = newr[:-1]
        self.ne[0][0] = newr[-1][0] / self.jacobian
        self.atomsx.set_positions(ratom)
        self._calc.set(nelect = self.ne[0][0])

    def __len__(self):
        return self.natom+1

    def get_forces(self,apply_constraint=True):
        f    = self.atomsx.get_forces(apply_constraint)
        self.get_mue()
        if self.e_only:
            Fc   = np.vstack((f*0.0, self.mue / self.jacobian))
        else:
            Fc   = np.vstack((f, self.mue / self.jacobian))
        return Fc
    
    def get_potential_energy(self, force_consistent=False):
        E0 = self.atomsx.get_potential_energy(force_consistent)

        # get the total number of electrons at neutral. 
        # should have been done in __ini__, but self._calc is not accessible there
        try: 
           self.n0
        except:
           # from ASE 3.22.0
           self.n0 = self._calc.default_nelect_from_ppp() # number of electrons at 0 charge
           print("zero charge electrons, n0:", self.n0)

        if self.ne[0][0] < 0.01: self.ne[0][0] = self._calc.get_number_of_electrons()
        E0 += (self.ne[0][0]-self.n0) * (-self.epotential + self.vtot)
        return E0

    def get_mue(self):
        # the initial guess for ne is passed through the calculator
        self.ne[0][0] = self._calc.get_number_of_electrons()

        # aligh the vacuum level and get the potential shift due to the compensating charge
        # vacuumE is used when the charge is compensated in solvent by VSAPsol
        # vtot_new is integrated when the charge is compensated by uniform background charge
        vacuumE, vtot_new = align_vacuum(direction = self.direction, LOCPOTfile='LOCPOT')

        if self.solPoisson:
            self.vtot = -vacuumE
        else:
            # start from zero charge for absolute reference
            try: self.vtot0 += 0 # if vtot0 exists, do nothing
            except: self.vtot0 = vtot_new # save the first vtot_new as vtot0
            self.vtot = (self.vtot0 + vtot_new)*0.5

        self.mue[0][0]  = self.epotential - (self._calc.get_fermi_level() - vacuumE)
        print("mu of electron: ", self.mue)
        print("number of electrons: ", self.ne)


    def copy(self):
        """Return a copy."""
        import copy
        atomsy = self.atomsx.copy()
        atoms = self.__class__(atomsy, self.epotential)

        atoms.arrays = {}
        for name, a in self.arrays.items():
            atoms.arrays[name] = a.copy()
        #atoms.constraints = copy.deepcopy(self.constraints)
        #atoms.adsorbate_info = copy.deepcopy(self.adsorbate_info)
        return atoms

