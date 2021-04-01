"""
The (electrochemical) nudged elastic path, (e)neb, module.
Contact: Penghao Xiao (pxiao@utexas.edu)
Version: 1.0
Reference: coming soon
Usage: move the slab to the center of the box first, as in eAtoms.py

NEB under constant electrochemical potential, U, where number of electrons is 
an additional variable 

When "parallel=True" is set, (e)neb is parallelized over images through mpi4py.
Each image can only use one processor, because the MPI comminicator cannot be 
passed to the calculator. The command to run the neb script should look like:

mpirun -np N python filename.py

where N equals the number of intermedia images, excluding the two end points.

The other parallel version of (e)neb, pssneb.py, parallelizes over images through
python pool and then each image invokes mpirun when calling the calculator. 
pssneb has only been tested for the VASP calculator.
"""


import numpy
import os,sys
from copy import deepcopy
from math import sqrt, atan, pi
from util import vmag, vunit, vproj, vdot, sPBC, vmag2
from ase import atoms, units
from ase.calculators.vasp import VaspChargeDensity

class eneb:
    """
    The generalized nudged elastic path (eneb) class.
    """

    def __init__(self, p1, p2, numImages = 7, k = 5.0, tangent = "new",       \
                 dneb = False, dnebOrg = False, method = 'normal',            \
                 onlyci = False, weight = 1, parallel = False, ss = False,    \
                 eneb = True, epotential = 0.0, ne1=0.0, ne2=0.0, ne0=0.0, eweight=0.0, solPoisson=True,\
                 express = numpy.zeros((3,3)), fixstrain = numpy.ones((3,3)) ):
        """
        The neb constructor.
        Parameters:
            p1.......... one endpoint of the path
            p2.......... the other endpoint of the path
            numImages... the total number of images in the path, including the 
                         endpoints
            k........... the spring force constant
            tangent..... the tangent method to use, "new" for the new tangent,
                         anything else for the old tangent
            dneb........ set to true to use the double-nudging method
            dnebOrg..... set to true to use the original double-nudging method
            method...... "ci" for the climbing image method, anything else for
                         normal NEB method 
            ss........   boolean, cell change or not
            express..... external press, 3*3 lower triangular matrix in the 
                         unit of GPa
            fixstrain... 3*3 matrix as express. 
            eneb........ boolean, number of electrons change  or not 
            epotential.. electrochemical potential: the work function of the conter electrode under the given potential
            ne0......... number of electrons at zero charge
            ne1......... number of electrons for the initial state at epotential
            ne2......... number of electrons for the final state at epotential
            vtot........ average Coulomb potential, to compensate the background charge in VASP
            solPoisson.. True corresponds to setting the compensate charge in the solvent, where VASPsol is required with lambda_d_k=3.0; 
                         False corresponds to using the uniform background charge and the correction scheme from the Neurock group;
        """

        self.numImages = numImages
        self.k         = k * numImages
        self.tangent   = tangent
        self.dneb      = dneb
        self.dnebOrg   = dnebOrg
        self.method    = method
        self.onlyci    = onlyci
        self.weight    = weight 
        self.parallel  = parallel 
        self.ss        = ss
        self.express   = express * units.GPa
        if express[0][1]**2+express[0][2]**2+express[1][2]**2 > 1e-3:
           express[0][1] = 0
           express[0][2] = 0
           express[1][2] = 0
           if (not self.parallel) or (self.parallel and self.rank == 0):
               print("warning: xy, xz, yz components of the external pressure will be set to zero")
        self.fixstrain = fixstrain
        self.eneb      = eneb
        self.epotential= -epotential-4.6
        self.ne0 = ne0 # number of electrons at zero charge
        self.eweight = eweight
        self.solPoisson = solPoisson
        self.direction = 'z' # slab norm direction

        # check the orientation of the cell, make sure a is along x, b is on xoy plane
        if self.ss: 
          for p in [p1,p2]:
            cr = p.get_cell()
            if cr[0][1]**2+cr[0][2]**2+cr[1][2]**2 > 1e-3: 
                if (not self.parallel) or (self.parallel and self.rank == 0):
                    print("check the orientation of the cell, make sure a is along x, b is on the x-y plane")
                sys.exit()

        # parallel over images through mpi4py
        if self.parallel:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.size = self.comm.size
            self.rank = self.comm.rank
            self.MPIDB= MPI.DOUBLE

        # set the path by linear interpolation between end points
        n = self.numImages - 1
        self.path = [p1]
        self.path+= [p1.copy() for i in range(self.numImages-2)]
        self.path+= [p2]
        cell1 = p1.get_cell()
        cell2 = p2.get_cell()
        dRB   = (cell2 - cell1) / n # path for cell
        # don't use get_scaled_positions() or apply sPBC() here
        # if the atoms can move over half of the lattice from initial to final
        icell = numpy.linalg.inv(cell1)
        vdir1 = numpy.dot(p1.get_positions(),icell)
        icell = numpy.linalg.inv(cell2)
        vdir2 = numpy.dot(p2.get_positions(),icell)
        dR    = sPBC(vdir2 - vdir1) / n # path for direct coordinates
        calc  = p1.get_calculator()
        # electrochemical or not
        if self.eneb:
            if ne0 == 0: 
                print("Please set the total number electrons at zero charge, ne0")
                sys.exit()
            if ne2 == 0 or ne1 == 0: 
                print("Please set the total number electrons in initial (ne1) and final (ne2) states at the desired voltage (epotential)")
                sys.exit()
            dne = (ne2-ne1) / (n + 0.0)
        for i in range(1, n):
            # making a directory for each image, which is nessecary for vasp to read last step's WAVECAR
            # also, it is good to prevent overwriting files for parallelizaiton over images
            fdname = '0'+str(i)
            if (not self.parallel) or (self.parallel and self.rank == 0):
                if not os.path.exists(fdname): os.mkdir(fdname)
            cellt = cell1 + dRB * i
            vdirt = vdir1 + dR * i
            rt    = numpy.dot(vdirt,cellt)
            self.path[i].set_cell(cellt)
            self.path[i].set_positions(rt)
            self.path[i].set_calculator(calc)
            self.path[i].st = numpy.zeros((3,3))
            # electrochemical or not
            if self.eneb:
                self.path[i].ne = ne1 + dne * i
                self.path[i].mue = numpy.zeros((1,3))
        self.Umaxi = 1

        # calculate the Jacobian so that a cell move have the same units and weight as an atomic move
        vol1     = self.path[0].get_volume()
        vol2     = self.path[self.numImages-1].get_volume()
        vol      = (vol1+vol2)*0.5
        self.natom = len(self.path[0]) 
        avglen   = (vol/self.natom)**(1.0/3.0)
        self.jacobian = avglen * self.natom**0.5 * self.weight

        # add some new properties
        for i in [0,n]:
            fdname = '0'+str(i)
            backfd = '../'
            if self.parallel: 
                fdname += '/'+str(self.rank)+str(i)
                backfd  = '../../'
            if not os.path.exists(fdname): os.makedirs(fdname)
            os.chdir(fdname)
            # electrochemical 
            if self.eneb:
                self.path[i].ne = ne1 + dne * i
                self.path[i]._calc.set(nelect = self.path[i].ne)
                print(i, self.path[i].ne)
            self.path[i].u = self.path[i].get_potential_energy()
            self.path[i].f = self.path[i].get_forces()
            if self.ss: stt = self.path[i].get_stress()
            if self.eneb: 
                self.path[i].mue = numpy.zeros((1,3))
                self.get_vtot(i)
                # calculate the eU term and energy correction in electrochemical constant voltage setting
                self.path[i].u += (self.path[i].ne-self.ne0) * (-self.epotential + self.path[i].vtot)
            os.chdir(backfd)
            self.path[i].cellt = self.path[i].get_cell() * self.jacobian 
            self.path[i].icell = numpy.linalg.inv(self.path[i].get_cell())
            self.path[i].vdir  = self.path[i].get_scaled_positions()
            self.path[i].st = numpy.zeros((3,3))
            # solid-state or not
            if self.ss:
                vol = self.path[i].get_volume()*(-1)
                self.path[i].st[0][0] = stt[0] * vol
                self.path[i].st[1][1] = stt[1] * vol
                self.path[i].st[2][2] = stt[2] * vol
                self.path[i].st[2][1] = stt[3] * vol
                self.path[i].st[2][0] = stt[4] * vol
                self.path[i].st[1][0] = stt[5] * vol
                self.path[i].st      -= self.express * (-1)*vol
                self.path[i].st      *= self.fixstrain 

            # calculate the PV term in the enthalpy E+PV, setting image 0 as reference
            dcell  = self.path[i].get_cell() - self.path[0].get_cell()
            strain = numpy.dot(self.path[0].icell, dcell)
            pv     = numpy.vdot(self.express, strain) * self.path[0].get_volume()
            #if (not self.parallel) or (self.parallel and self.rank == 0):
                #print "i,pv:",i,pv
            self.path[i].u += pv

    def get_vtot(self, imgi):
        #self.path[imgi].ne = self.path[imgi]._calc.get_number_of_electrons()
        # the following part is adjusted from vtotav.py in ase_tools
        # https://github.com/compphys/ase_tools/blob/master/scripts/vtotav.py
        # First specify location of LOCPOT 
        LOCPOTfile = 'LOCPOT'
        # Next the direction to make average in 
        # input should be x y z, or X Y Z. Default is Z.
        allowed = "xyzXYZ"
        if allowed.find(self.direction) == -1 or len(self.direction)!=1 :
           print("** WARNING: The direction was input incorrectly.")
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
        potl=potl*self.path[imgi].get_volume()

        print("\nReading file: %s" % LOCPOTfile)
        print("Performing average in %s direction" % self.direction)

        # lattice parameters and scale factor
        #---------------------------------------------
        cell = self.path[imgi].cell

        # Find length of lattice vectors
        #--------------------------------
        latticelength = numpy.dot(cell, cell.T).diagonal()
        latticelength = latticelength**0.5

        # Read in potential data
        #------------------------
        ngridpts = numpy.array(potl.shape)
        totgridpts = ngridpts.prod()
        #print "Potential stored on a %dx%dx%d grid" % (ngridpts[0],ngridpts[1],ngridpts[2])
        #print "Total number of points is %d" % totgridpts
        #print "Reading potential data from file...",
        #sys.stdout.flush()
        #print "done." 

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
        average = numpy.zeros(ngridpts[idir],numpy.float)
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
        #print "Writing averaged data to file %s..." % averagefile,
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
            self.path[imgi].vtot = -vacuumE
        else:
            # assume vtot0 at zero charge have similar magnitudes along the path,
            # then for relative energy difference, vtot0 can be ignored
            self.path[imgi].vtot = numpy.average(average-vacuumE)*0.5
            # xph: integrate from zero charge for an absolute energy reference
            # need post processing to calculate the vtot0 for the relaxed structure
            #vtot_new = numpy.average(average-vacuumE)
            #self.path[imagi].vtot = (vtot_new + self.path[imagi].vtot0)*0.5
        self.path[imgi].mue[0][0]  = self.epotential - (self.path[imgi]._calc.get_fermi_level() - vacuumE)
        print("image {}, mu of electron: {}, number of electrons: {} ".format(imgi, self.path[imgi].mue[0][0], self.path[imgi].ne))
        #print "done."

    def forces(self):
        """
        Calculate the forces for each image on the path.  Applies the force due
        to the potential and the spring forces.
        Parameters:
            force - the potential energy force.
        """

        # Calculate the force due to the potential on the intermediate points
        
        #=========================== Begin potential energy evaluation ==============================
        #--------------------------- MPI version -------------------------
        if self.parallel:
            imgi  = self.rank+1
            fdname = '0'+str(imgi)
            os.chdir(fdname)
            self.path[imgi].u    = self.path[imgi].get_potential_energy()
            self.path[imgi].f    = self.path[imgi].get_forces()
            if self.ss: stt      = self.path[imgi].get_stress()
            os.chdir('../')

            try:
                self.path[imgi].st
            except:
                self.path[imgi].st = numpy.zeros((3,3))
            # solid-state or not
            if self.ss:
                vol = self.path[imgi].get_volume()*(-1)
                self.path[imgi].st[0][0] = stt[0] * vol
                self.path[imgi].st[1][1] = stt[1] * vol
                self.path[imgi].st[2][2] = stt[2] * vol
                self.path[imgi].st[2][1] = stt[3] * vol
                self.path[imgi].st[2][0] = stt[4] * vol
                self.path[imgi].st[1][0] = stt[5] * vol
                self.path[imgi].st[0][1] = 0.0
                self.path[imgi].st[0][2] = 0.0
                self.path[imgi].st[1][2] = 0.0
                self.path[imgi].st      -= self.express * vol*(-1)
                self.path[imgi].st      *= self.fixstrain 

            ui    = self.path[imgi].u 
            fi    = self.path[imgi].f 
            sti   = self.path[imgi].st 
            msg_s = numpy.vstack((fi, sti, [ui,0.0,0.0]))
            msg_r = numpy.zeros((self.size, self.natom+4,3))

            #The following pypar send and receive are equivalent to Allgather()
            #msg_r=pypar.gather(msg_s,0,buffer=msg_r)
            #msg_r=pypar.broadcast(msg_r,0)
            self.comm.Allgather([msg_s, self.MPIDB], [msg_r, self.MPIDB])

            for i in range(1, self.numImages - 1):
                self.path[i].f = msg_r[i-1][:-4]
                self.path[i].st = msg_r[i-1][-4:-1]
                self.path[i].u = msg_r[i-1][-1][0]
        #--------------------------- Serial version -------------------------
        else: 
            for i in range(1, self.numImages - 1):
                # writing input and do the calculation in images' directories respectively
                fdname = '0'+str(i)
                os.chdir(fdname)
                # electrochemical 
                if self.eneb:
                    self.path[i]._calc.set(nelect = self.path[i].ne)
                self.path[i].u     = self.path[i].get_potential_energy()
                self.path[i].f     = self.path[i].get_forces()
                if self.ss: stt    = self.path[i].get_stress()
                if self.eneb: 
                    self.get_vtot(i)
                    # calculate the eU term and energy correction in electrochemical constant voltage setting
                    self.path[i].u += (self.path[i].ne-self.ne0) * (-self.epotential + self.path[i].vtot)
                os.chdir('../')
                # solid-state or not
                if self.ss:
                    vol = self.path[i].get_volume()*(-1)
                    self.path[i].st[0][0] = stt[0] * vol
                    self.path[i].st[1][1] = stt[1] * vol
                    self.path[i].st[2][2] = stt[2] * vol
                    self.path[i].st[2][1] = stt[3] * vol
                    self.path[i].st[2][0] = stt[4] * vol
                    self.path[i].st[1][0] = stt[5] * vol
                    self.path[i].st[0][1] = 0.0
                    self.path[i].st[0][2] = 0.0
                    self.path[i].st[1][2] = 0.0
                    self.path[i].st      -= self.express * vol*(-1)
                    self.path[i].st      *= self.fixstrain 
        #=========================== End potential energy evaluation ==============================

        for i in range(1, self.numImages - 1):
            self.path[i].cellt = self.path[i].get_cell() * self.jacobian 
            self.path[i].icell = numpy.linalg.inv(self.path[i].get_cell())
            self.path[i].vdir  = self.path[i].get_scaled_positions()

            # calculate the PV term in the enthalpy E+PV, setting image 0 as reference
            dcell  = self.path[i].get_cell() - self.path[0].get_cell()
            strain = numpy.dot(self.path[0].icell, dcell)
            pv     = numpy.vdot(self.express, strain) * self.path[0].get_volume()
            if (not self.parallel) or (self.parallel and self.rank == 0):
                #print "i,pv:",i,pv
                print("i,mue:", i, self.path[i].mue)
            self.path[i].u += pv

            if i == 1 or self.path[i].u > self.Umax:
                self.Umax  = self.path[i].u
                self.Umaxi = i
            
        # Loop over each intermediate point and calculate the tangent.
        for i in range(1, self.numImages - 1):

            # Here st should be the Cauchy stress tensor times cell volume. 
            # Timing box volume should have been done.
            self.path[i].totalf = numpy.vstack((self.path[i].f, self.path[i].st / self.jacobian))
            # electrochemical 
            if self.eneb:
                 tmpw = self.eweight if self.eweight else 1.0
                 self.path[i].totalf = numpy.vstack((self.path[i].totalf, self.path[i].mue / tmpw))
            # realtf that needed by nebspline.pl is saved for output
            self.path[i].realtf = deepcopy(self.path[i].totalf)
            
            # If we're using the 'old' tangent, the tangent is defined as the
            # vector from the point behind the current image to the point in
            # front of the current image.
            # Haven't implemented for ssneb
            if self.tangent == 'old':
                self.path[i].n = (self.path[i + 1].r - self.path[i - 1].r)
            
            # Otherwise, we're using the 'new' tangent.
            # Ref:
            # G. Henkelman and H. Jonsson,  Improved tangent estimate in the 
            # nudged elastic path method for finding minimum energy paths and 
            # saddle points, J. Chem. Phys. 113, 9978-9985 (2000)
            else:
                # UPm1: is the previous image higher in energy
                # UPp1: is the next image higher in energy
                UPm1 = self.path[i - 1].u > self.path[i].u
                UPp1 = self.path[i + 1].u > self.path[i].u
                
                # if V(i+1)>V(i)>V(i-1)
                # or V(i+1)<V(i)<V(i-1)
                # (this is the usual case along the MEP)
                '''
                tangent
                '''
                if(UPm1 != UPp1):
                    if(UPm1):
                        # use direct coordinates to avoid double counting cell motion
                        dr_dir  = sPBC(self.path[i].vdir - self.path[i - 1].vdir)
                        avgbox  = 0.5*(self.path[i].get_cell() + self.path[i - 1].get_cell())
                        sn  = numpy.dot(dr_dir,avgbox)
                        dh  = self.path[i].cellt - self.path[i - 1].cellt
                        snb = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i - 1].icell, dh)*0.5
                        #---------------another way to average strain----------------------
                        #iavgbox = numpy.linalg.inv(avgbox)
                        #snb = numpy.dot(iavgbox, snb)
                        #------------------------------------------------------------------
                        self.path[i].n = numpy.vstack((sn,snb))
                        # electrochemical 
                        if self.eneb:
                            sne = numpy.zeros((1,3))
                            sne[0][0] = (self.path[i].ne - self.path[i - 1].ne) * self.eweight
                            self.path[i].n = numpy.vstack((self.path[i].n, sne))
                    else:
                        dr_dir  = sPBC(self.path[i + 1].vdir - self.path[i].vdir)
                        avgbox  = 0.5*(self.path[i+1].get_cell() + self.path[i].get_cell())
                        sn  = numpy.dot(dr_dir,avgbox)
                        dh  = self.path[i + 1].cellt - self.path[i].cellt
                        snb = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i + 1].icell, dh)*0.5
                        #---------------another way to average strain----------------------
                        #iavgbox = numpy.linalg.inv(avgbox)
                        #snb = numpy.dot(iavgbox, snb)
                        #------------------------------------------------------------------
                        self.path[i].n = numpy.vstack((sn,snb))
                        # electrochemical 
                        if self.eneb:
                            sne = numpy.zeros((1,3))
                            sne[0][0] = (self.path[i + 1].ne - self.path[i].ne) * self.eweight
                            self.path[i].n = numpy.vstack((self.path[i].n, sne))
                # otherwise, we are near some extremum
                else:
                    Um1 = self.path[i - 1].u - self.path[i].u
                    Up1 = self.path[i + 1].u - self.path[i].u
                    Umin = min(abs(Up1), abs(Um1))
                    Umax = max(abs(Up1), abs(Um1))
                    if(Um1 > Up1):
                        dr_dir  = sPBC(self.path[i + 1].vdir - self.path[i].vdir)
                        avgbox  = 0.5*(self.path[i + 1].get_cell() + self.path[i].get_cell())
                        sn      = numpy.dot(dr_dir,avgbox) * Umin
                        dr_dir  = sPBC(self.path[i].vdir - self.path[i - 1].vdir)
                        avgbox  = 0.5*(self.path[i].get_cell() + self.path[i - 1].get_cell())
                        sn     += numpy.dot(dr_dir,avgbox) * Umax

                        dh   = self.path[i + 1].cellt - self.path[i].cellt
                        snb1 = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i + 1].icell, dh)*0.5
                        dh   = self.path[i].cellt - self.path[i - 1].cellt
                        snb2 = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i - 1].icell, dh)*0.5
                        snb  = snb1 * Umin + snb2 * Umax
                        self.path[i].n = numpy.vstack((sn,snb))
                        # electrochemical 
                        if self.eneb:
                            sne1 = numpy.zeros((1,3))
                            sne1[0][0] = (self.path[i + 1].ne - self.path[i].ne) * self.eweight
                            sne2 = numpy.zeros((1,3))
                            sne2[0][0] = (self.path[i].ne - self.path[i - 1].ne) * self.eweight
                            sne  = sne1 * Umin + sne2 * Umax
                            self.path[i].n = numpy.vstack((self.path[i].n, sne))
                    else:
                        dr_dir  = sPBC(self.path[i + 1].vdir - self.path[i].vdir)
                        avgbox  = 0.5*(self.path[i + 1].get_cell() + self.path[i].get_cell())
                        sn      = numpy.dot(dr_dir,avgbox) * Umax
                        dr_dir  = sPBC(self.path[i].vdir - self.path[i - 1].vdir)
                        avgbox  = 0.5*(self.path[i].get_cell() + self.path[i - 1].get_cell())
                        sn     += numpy.dot(dr_dir,avgbox) * Umin

                        dh   = self.path[i + 1].cellt - self.path[i].cellt
                        snb1 = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i + 1].icell, dh)*0.5
                        dh   = self.path[i].cellt - self.path[i - 1].cellt
                        snb2 = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i - 1].icell, dh)*0.5
                        snb  = snb1 * Umax + snb2 * Umin
                        self.path[i].n = numpy.vstack((sn,snb))
                        # electrochemical 
                        if self.eneb:
                            sne1 = numpy.zeros((1,3))
                            sne1[0][0] = (self.path[i + 1].ne - self.path[i].ne) * self.eweight
                            sne2 = numpy.zeros((1,3))
                            sne2[0][0] = (self.path[i].ne - self.path[i - 1].ne) * self.eweight
                            sne  = sne1 * Umax + sne2 * Umin
                            self.path[i].n = numpy.vstack((self.path[i].n,sne))

        # Normalize each tangent
        if (not self.parallel) or (self.parallel and self.rank == 0):
            print("==========!tangent contribution!==========")
            print("Jacobian:", self.jacobian)
            print("ImageNum        atom         cell")
        for i in range(1,self.numImages-1):
            self.path[i].n = vunit(self.path[i].n)
            if (not self.parallel) or (self.parallel and self.rank == 0):
                print(i, vmag(self.path[i].n[:-3]), vmag(self.path[i].n[-3:]))

        # Loop over each intermediate image and adjust the potential energy,
        # force, and apply the spring force.
        for i in range(1, self.numImages - 1):

            # push the climbing image uphill
            if self.method == 'ci' and i == self.Umaxi:
                self.path[i].totalf -= 2.0 * vproj(self.path[i].totalf, self.path[i].n) 
                self.path[i].fPerp   = self.path[i].totalf

            # and for the non-climbing images...
            else:

                # Calculate the force perpendicular to the tangent. 
                self.path[i].fPerp = self.path[i].totalf - vproj(self.path[i].totalf,   \
                                                            self.path[i].n)
                # Calculate the spring force.
                Rm1  = sPBC(self.path[i - 1].vdir - self.path[i].vdir)
                avgbox  = 0.5*(self.path[i - 1].get_cell() + self.path[i].get_cell())
                Rm1  = numpy.dot(Rm1,avgbox) 
                dh   = self.path[i - 1].cellt - self.path[i].cellt
                Rm1b = numpy.dot(self.path[i].icell, dh)*0.5 + numpy.dot(self.path[i - 1].icell, dh)*0.5
                Rm1  = numpy.vstack((Rm1,Rm1b))
                # electrochemical 
                if self.eneb:
                    Rm1e = numpy.zeros((1,3))
                    Rm1e[0][0] = (self.path[i - 1].ne - self.path[i].ne) * self.eweight
                    Rm1  = numpy.vstack((Rm1,Rm1e))

                Rp1  = sPBC(self.path[i + 1].vdir - self.path[i].vdir)
                avgbox  = 0.5*(self.path[i + 1].get_cell() + self.path[i].get_cell())
                Rp1  = numpy.dot(Rp1,avgbox)
                dh   = self.path[i + 1].cellt - self.path[i].cellt
                Rp1b = numpy.dot(self.path[i].icell, dh)*0.5+numpy.dot(self.path[i + 1].icell, dh)*0.5
                Rp1  = numpy.vstack((Rp1,Rp1b))
                # electrochemical 
                if self.eneb:
                    Rp1e = numpy.zeros((1,3))
                    Rp1e[0][0] = (self.path[i + 1].ne - self.path[i].ne) * self.eweight
                    Rp1  = numpy.vstack((Rp1,Rp1e))

                self.path[i].fsN = (vmag(Rp1) - vmag(Rm1)) * self.k * self.path[i].n
                #print i, vmag(Rp1),vmag(Rm1)

                # For dneb use total spring force -spring force in the grad direction.
                if self.dneb:
                    self.path[i].fs = (Rp1 + Rm1) * self.k
                    self.path[i].fsperp = self.path[i].fs -                   \
                                          vproj(self.path[i].fs, self.path[i].n)
                    self.path[i].fsdneb = self.path[i].fsperp -               \
                                          vproj(self.path[i].fs, self.path[i].fPerp)

                    # dneb modification so that it will converge
                    if not self.dnebOrg:
                        FperpSQ = vmag2(self.path[i].fPerp)
                        FsperpSQ = vmag2(self.path[i].fsperp)
                        if FsperpSQ > 0:
                            self.path[i].fsdneb *= 2.0/pi*atan(FperpSQ/FsperpSQ)

                # Not using double-nudging, so set the double-nudging spring
                # force to zero.
                else:
                    self.path[i].fsdneb = 0

                # The final force is the sum of these forces.    
                self.path[i].totalf = self.path[i].fsdneb + self.path[i].fsN +     \
                                 self.path[i].fPerp

                # only move the climing image
                if(self.method == 'ci' and self.onlyci): 
                    self.path[i].totalf *= 0.0

