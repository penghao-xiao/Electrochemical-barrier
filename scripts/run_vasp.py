# set this script and the POTCAR files in your path
# see https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html
import os
exitcode = os.system('srun -c 1  /usr/gapps/qsg/VASP/bin/vasp_dev_quartz5_5.4.4_TST_new_xray_patched_sol')
