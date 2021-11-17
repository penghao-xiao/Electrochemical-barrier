'''
the VaspoLocpot class is contributed by Nathan Keilbart
https://gitlab.com/nathankeilbart/ase/-/blob/VaspLocpot/ase/calculators/vasp/vasp_auxiliary.py
'''
from ase import Atoms
import numpy as np
from typing import Optional

class VaspLocpot:
    """Class for reading the Locpot VASP file.

    Filename is normally LOCPOT.

    Coding is borrowed from the VaspChargeDensity class and altered to work for LOCPOT."""
    def __init__(self, atoms: Atoms, pot: np.ndarray, 
                 spin_down_pot: Optional[np.ndarray] = None,
                 magmom: Optional[np.ndarray] = None) -> None:
        self.atoms = atoms
        self.pot = pot
        self.spin_down_pot = spin_down_pot
        self.magmom = magmom

    @staticmethod
    def _read_pot(fobj, pot):
        """Read potential from file object

        Utility method for reading the actual potential from a file object. 
        On input, the file object must be at the beginning of the charge block, on
        output the file position will be left at the end of the
        block. The pot array must be of the correct dimensions.

        """
        # VASP writes charge density as
        # WRITE(IU,FORM) (((C(NX,NY,NZ),NX=1,NGXC),NY=1,NGYZ),NZ=1,NGZC)
        # Fortran nested implied do loops; innermost index fastest
        # First, just read it in
        for zz in range(pot.shape[2]):
            for yy in range(pot.shape[1]):
                pot[:, yy, zz] = np.fromfile(fobj, count=pot.shape[0], sep=' ')

    @classmethod
    def from_file(cls, filename='LOCPOT'):
        """Read LOCPOT file.

        LOCPOT contains local potential.

        Currently will check for a spin-up and spin-down component but has not been
        configured for a noncollinear calculation.

        """
        import ase.io.vasp as aiv
        with open(filename,'r') as fd:
            try:
                atoms = aiv.read_vasp(fd)
            except (IOError, ValueError, IndexError):
                return print('Error reading in initial atomic structure.')
            fd.readline()
            ngr = fd.readline().split()
            ng = (int(ngr[0]), int(ngr[1]), int(ngr[2]))
            pot = np.empty(ng)
            cls._read_pot(fd, pot)
            # Check if the file has a spin-polarized local potential, and
            # if so, read it in.
            fl = fd.tell()
            # Check to see if there is more information
            line1 = fd.readline()
            if line1 == '':
                return cls(atoms,pot)
            # Check to see if the next line equals the previous grid settings
            elif line1.split() == ngr:
                spin_down_pot = np.empty(ng)
                cls._read_pot(fd, spin_down_pot)
            elif line1.split() != ngr:
                fd.seek(fl)
                magmom = np.fromfile(fd, count=len(atoms), sep=' ')
                line1 = fd.readline()
                if line1.split() == ngr:
                    spin_down_pot = np.empty(ng)
                    cls._read_pot(fd, spin_down_pot)
        fd.close()
        return cls(atoms, pot, spin_down_pot=spin_down_pot, magmom=magmom)

    def get_average_along_axis(self,axis=2,spin_down=False):
        """
        Returns the average potential along the specified axis (0,1,2).

        axis: Which axis to average long
        spin_down: Whether to use the spin_down_pot instead of pot
        """
        if axis not in [0,1,2]:
            return print('Must provide an integer value of 0, 1, or 2.')
        average = []
        if spin_down:
            pot = self.spin_down_pot
        else:
            pot = self.pot
        if axis == 0:
            for i in range(pot.shape[axis]):
                average.append(np.average(pot[i,:,:]))
        elif axis == 1:
            for i in range(pot.shape[axis]):
                average.append(np.average(pot[:,i,:]))
        elif axis == 2:
            for i in range(pot.shape[axis]):
                average.append(np.average(pot[:,:,i]))
        return average

    def distance_along_axis(self,axis=2):
        """
        Returns the scalar distance along axis (from 0 to 1).
        """
        if axis not in [0,1,2]:
            return print('Must provide an integer value of 0, 1, or 2.')
        return np.linspace(0,1,self.pot.shape[axis],endpoint=False)

    def is_spin_polarized(self):
        return (self.spin_down_pot is not None)

def align_vacuum(direction='Z', LOCPOTfile='LOCPOT'):
        
    '''
    aligh the vacuum level to the avg LOCPOT at the end of the simulation box
    (make sure it is vacuum there)
    returns: 
         the vacuum level ()
         the average electrostatic potential (vtot_new)
    '''
    # the direction to make average in 
    # input should be x y z, or X Y Z. Default is Z.
    allowed = "xyzXYZ"
    if allowed.find(direction) == -1 or len(direction)!=1 :
       print("** WARNING: The direction was input incorrectly." )
       print("** Setting to z-direction by default.")
    if direction.islower():
       direction = direction.upper()
    filesuffix = "_%s" % direction

    # Open geometry and density class objects
    #-----------------------------------------
    axis_translate = {'X':0, 'Y':1, 'Z':2}
    ax = axis_translate[direction]
    #vasp_charge = VaspChargeDensity(filename = LOCPOTfile)
    vasp_locpot = VaspLocpot.from_file(filename = LOCPOTfile)
    average = vasp_locpot.get_average_along_axis(axis=ax)
    average = np.array(average)

    # lattice parameters and scale factor
    #---------------------------------------------
    cell = vasp_locpot.atoms.cell
    # Find length of lattice vectors
    #--------------------------------
    latticelength = np.dot(cell, cell.T).diagonal()
    latticelength = latticelength**0.5
    # Print out average
    #-------------------
    averagefile = LOCPOTfile + filesuffix
    #print("Writing averaged data to file %s..." % averagefile,)
    #sys.stdout.flush()
    outputfile = open(averagefile,"w")
    outputfile.write("#  Distance(Ang)     Potential(eV)\n")
    xdis = vasp_locpot.distance_along_axis(axis=ax) * latticelength[ax]
    for xi, poti in zip(xdis, average):
       outputfile.write("%15.8g %15.8g\n" % (xi,poti))
    outputfile.close()
    del vasp_locpot

    vacuumE = average[-1]
    vtot_new = np.average(average-vacuumE)

    return vacuumE, vtot_new 

