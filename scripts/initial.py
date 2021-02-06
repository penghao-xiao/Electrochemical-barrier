#!/usr/bin/env python

'''
Neb optimization example
'''

from ase import io
import numpy, copy, os

def sPBC(vdir):
    return (vdir % 1.0 + 1.5) % 1.0 - 0.5

numImages = 7
i     = 0
j     = 6
file1 = str(i)+'.CON'
file2 = str(j)+'.CON'
p1 = io.read(file1,format='vasp')
p2 = io.read(file2,format='vasp')

dRB   = (p2.get_cell() - p1.get_cell()) / (numImages - 1.0) 
ibox  = numpy.linalg.inv(p1.get_cell())
vdir1 = numpy.dot(p1.get_positions(),ibox)
ibox  = numpy.linalg.inv(p2.get_cell())
vdir2 = numpy.dot(p2.get_positions(),ibox)
dR    = sPBC(vdir2 - vdir1) / (numImages - 1.0) 

for k in range(1,numImages-1):
    pm   = copy.deepcopy(p1)
    box  = pm.get_cell()
    box += dRB*k
    vdir = vdir1 + dR*k
    r    = numpy.dot(vdir,box)
    pm.set_cell(box)
    pm.set_positions(r)
    f3 = str(k) + '.CON'
    io.write(f3,pm,format='vasp')

for k in range(numImages):
    fdname = '0'+str(k)
    if not os.path.exists(fdname): os.mkdir(fdname)
    os.system('cat '+str(k)+'.CON >>movie.con')
    os.system('cp '+str(k)+'.CON '+fdname+'/POSCAR')
