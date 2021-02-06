from ase.io import read, write
import numpy as np
from numpy import linalg
from ase.constraints import FixAtoms

file = "POS0.vasp"
p = read(file, format= 'vasp')
r = p.get_positions()
cell = p.get_cell()
r0 = [0, 0, 0.5]
r0 = np.dot(r0, cell)
dr = r0 
for rtmp in r:
    rtmp += dr
p.set_positions(r)
#constraint = FixAtoms(indices=[atom.index for atom in p if atom.position[2] < r0])
#p.set_constraint(constraint)
write('POS1.vasp', p, format='vasp', direct = True, vasp5=True)


