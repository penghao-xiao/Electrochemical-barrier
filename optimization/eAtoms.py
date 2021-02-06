#!/usr/bin/env python
'''
Generalized Atoms calss for optimization of geometry and number of electrons simultaneously under constant voltage
Author: Penghao Xiao (pxiao@utexas.edu)
Version: 1.0
Usage: first set the vacuum/solution along z axis and move the slab to the center of the simulation box
Please cite the following reference:
'''

from ase import *
from ase.io import read,write
from ase import units
from ase.calculators.vasp import VaspChargeDensity
import numpy as np

class eAtoms(Atoms):
    def __init__(self, atomsx, n0, epotential=0.0, solPoisson=True):
        """ relaxation under constant electrochemical potential
            epotential ... electrochemical potential: the work function of the counter electrode under the given potential
                         i.e. potential vs. SHE + workfunction of SHE
            solPoisson.. True corresponds to compensate charge in the solvent, where VASPsol is required with lambda_d_k=3.0; 
                         False corresponds to uniform background charge;
        """
        self.atomsx = atomsx 
        self.epotential= -epotential - 4.6

        self.natom = atomsx.get_number_of_atoms()
        self.n0       = n0 # number of electrons at 0 charge
        self.ne       = np.zeros((1,3)) # number of electrons
        self.ne[0][0] = n0 
        self.mue      = np.zeros((1,3)) # electron mu, dE/dne
        self.vtot  = 0.0 # average electrostatic potential
        self.direction = 'z'
        self.solPoisson = solPoisson
        Atoms.__init__(self,atomsx)

    def get_positions(self):
        r    = self.atomsx.get_positions()
        #Rc   = np.concatenate((r, self.ne))
        Rc   = np.vstack((r, self.ne))
        return Rc

    def set_positions(self,newr):
        ratom   = newr[:-1]
        self.ne[0][0] = newr[-1][0]
        self.atomsx.set_positions(ratom)
        self._calc.set(nelect = self.ne[0][0])

    def __len__(self):
        return self.natom+1

    def get_forces(self,apply_constraint=True):
        f    = self.atomsx.get_forces(apply_constraint)
        #f    = self.atomsx.get_forces()
        self.get_vtot()
        #Fc   = np.concatenate((f, self.mue))
        Fc   = np.vstack((f, self.mue))
        return Fc
    
    def get_potential_energy(self, force_consistent=False):
        E0 = self.atomsx.get_potential_energy(force_consistent)
        E0 += (self.ne[0][0]-self.n0) * (-self.epotential + self.vtot)
        return E0

    def get_vtot(self):
        self.ne[0][0] = self._calc.get_number_of_electrons()
        # First specify location of LOCPOT 
        LOCPOTfile = 'LOCPOT'
        # Next the direction to make average in 
        # input should be x y z, or X Y Z. Default is Z.
        allowed = "xyzXYZ"
        if allowed.find(self.direction) == -1 or len(self.direction)!=1 :
           print("** WARNING: The direction was input incorrectly." )
           print("** Setting to z-direction by default.")
        if self.direction.islower():
           self.direction = self.direction.upper()
        filesuffix = "_%s" % self.direction

        # Open geometry and density class objects
        #-----------------------------------------
        vasp_charge = VaspChargeDensity(filename = LOCPOTfile)
        potl = vasp_charge.chg[-1]
        del vasp_charge

        # For LOCPOT files we multiply by the volume to get back to eV
        potl=potl*self.atomsx.get_volume()

        print("\nReading file: %s" % LOCPOTfile)
        print("Performing average in %s direction" % self.direction)

        # lattice parameters and scale factor
        #---------------------------------------------
        cell = self.atomsx.cell

        # Find length of lattice vectors
        #--------------------------------
        latticelength = np.dot(cell, cell.T).diagonal()
        latticelength = latticelength**0.5

        # Read in potential data
        #------------------------
        ngridpts = np.array(potl.shape)
        totgridpts = ngridpts.prod()
        #print("Potential stored on a %dx%dx%d grid" % (ngridpts[0],ngridpts[1],ngridpts[2]))
        #print("Total number of points is %d" % totgridpts)
        #print("Reading potential data from file...",)
        #sys.stdout.flush()
        #print("done." )

        # Perform average
        #-----------------
        if self.direction=="X":
           idir = 0
           a = 1
           b = 2
        elif self.direction=="Y":
           a = 0
           idir = 1
           b = 2
        else:
           a = 0
           b = 1
           idir = 2
        a = (idir+1)%3
        b = (idir+2)%3
        # At each point, sum over other two indices
        average = np.zeros(ngridpts[idir],np.float)
        for ipt in range(ngridpts[idir]):
           if self.direction=="X":
              average[ipt] = potl[ipt,:,:].sum()
           elif self.direction=="Y":
              average[ipt] = potl[:,ipt,:].sum()
           else:
              average[ipt] = potl[:,:,ipt].sum()

        # Scale by number of grid points in the plane.
        # The resulting unit will be eV.
        average /= ngridpts[a]*ngridpts[b]

        # Print out average
        #-------------------
        averagefile = LOCPOTfile + filesuffix
        #print("Writing averaged data to file %s..." % averagefile,)
        #sys.stdout.flush()
        outputfile = open(averagefile,"w")
        outputfile.write("#  Distance(Ang)     Potential(eV)\n")
        xdiff = latticelength[idir]/float(ngridpts[idir]-1)
        for i in range(ngridpts[idir]):
           x = i*xdiff
           outputfile.write("%15.8g %15.8g\n" % (x,average[i]))
        outputfile.close()
        vacuumE = average[-1]
        if self.solPoisson:
            self.vtot = -vacuumE
        else:
            self.vtot = numpy.average(average-vacuumE)
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

