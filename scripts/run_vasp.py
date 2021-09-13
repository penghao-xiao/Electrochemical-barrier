# set this script and the POTCAR files in your path
# see https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html
import os
exitcode = os.system('srun -c 1  path_to_vaspsol')
