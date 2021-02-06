# how to quickly set up structure to be compatible with approach:
from ase import io, atoms
 
atoms = io.read("POSCAR.vasp")
constraints = atoms.constraints
for i, spos in enumerate(atoms.get_scaled_positions()):
    if spos[-1] > 0.7 : 
         atoms[i].position -= atoms.get_cell()[-1]
 
atoms.center(axis=2)
atoms.set_constraint(constraints)
io.write('POSCAR.vasp_centered',atoms,direct=1)
