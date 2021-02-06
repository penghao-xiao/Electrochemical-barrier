
from util import vdot,vmag,vunit
from minimizer_essneb import minimizer_ssneb
import numpy as np

class fire_essneb(minimizer_ssneb):

    def __init__(self, path, maxmove = 0.2, dt = 0.1, dtmax = 1.0, 
                 Nmin = 5, finc = 1.1, fdec = 0.5, astart = 0.1, fa = 0.99):
        """
        path    - neb object to optimize

        Fire initializer function, called in script before min
        Optional arguments:
          dt:       initial dynamical timestep
          dtmax:    maximum timestep
          Nmin:     ???
          finc:     ???
          fdec:     ???
          astart:   ???
          fa:       ???
          maxmove:  maximum amount the point can move during optimization
        """

        minimizer_ssneb.__init__(self, path)
        self.maxmove=maxmove
        self.dt=dt
        self.dtmax=dtmax
        self.Nmin=Nmin
        self.finc=finc
        self.fdec=fdec
        self.astart = astart
        self.a=astart
        self.fa=fa
        self.Nsteps=0

        #self.v contains the velocity for both atoms and box for one image
        i = self.band.numImages-2
        #j = self.band.natom+3
        # electrochemical, add number of electrons in the variable vector
        if self.band.eneb:
            j = self.band.natom+4
        else:
            j = self.band.natom+3
        self.v = np.zeros((i,j,3))


    def step(self):
        """
        Fire step
        """
        self.band.forces()
        totalf = self.v.copy()
        for i in range(1, self.band.numImages - 1):
            totalf[i-1] = self.band.path[i].totalf
        Power = vdot(totalf,self.v)

        if Power > 0.0:
            self.v = (1.0-self.a)*self.v + self.a*vmag(self.v)*vunit(totalf)
            if(self.Nsteps>self.Nmin):
                self.dt = min(self.dt*self.finc,self.dtmax)
                self.a *= self.fa
            self.Nsteps += 1
        else:
            # reset velocity and slow down
            self.v  *= 0.0
            self.a   = self.astart
            self.dt *= self.fdec
            self.Nsteps  = 0

        # Euler step
        self.v += self.dt * totalf
        # check for max step
        #if vmag(self.v) > self.maxmove/self.dt :
        #    self.v = self.maxmove/self.dt * vunit(self.v)

        for i in range(1, self.band.numImages - 1):
            dR = self.dt * self.v[i-1]
            if vmag(dR) > self.maxmove:
                dR = self.maxmove * vunit(dR)
            # move R 
            rt  = self.band.path[i].get_positions()
            #rt += dR[:-3]
            # electrochemical, add number of electrons in the variable vector
            if self.band.eneb:
                rt += dR[:-4]
            else:
                rt += dR[:-3]
            self.band.path[i].set_positions(rt)

            # move box and update cartesian coordinates
            ct  = self.band.path[i].get_cell()
            #ct += np.dot(ct, dR[-3:]) / self.band.jacobian
            # electrochemical, add number of electrons in the variable vector
            if self.band.eneb:
                ct += np.dot(ct, dR[-4:-1]) / self.band.jacobian
            else:
                ct += np.dot(ct, dR[-3:]) / self.band.jacobian
            self.band.path[i].set_cell(ct, scale_atoms=True)

            # electrochemical, add number of electrons in the variable vector
            if self.band.eneb:
                tmpw = self.band.eweight if self.band.eweight else 1.0
                dne = dR[-1][0] / tmpw
                print("imagei and dne: ", i, dne)
                self.band.path[i].ne += dne

