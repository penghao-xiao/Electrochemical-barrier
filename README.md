# Electrochemical-barrier
This packages perform structure optimization and saddle search under constant potential
Contact: Penghao Xiao, pxiao@utexas.edu

Install:
 1. make sure VASPsol is compided
 2. pip3 install ASE
 3. set ASE with VASPsol as the binary, see script/run_vasp.py for the srun setting
    (potcar files and others, see https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#environment-variables)

Usage:
1. Before doing any calculation, first set the vacuum/solution along the z axis and move the slab to the center of the simulation box using scripts/center_slab.py (do not run the script separately for the initial and final states because there could be some small mismatch for the substrate.)
2. Optimize the end points with optimization/constV_opt.py
   grep FIRE slurm... to check the magnitude of residual forces (force, mu_e)
   grep electron slurm... for number of electrons and mu_e
3. copy the initial and final structures to 0.CON and 6.CON for an eNEB run
3. set the optimized number of electrons for the initial and fianl states in ne1=... and ne2=... in example/run_eneb-vasp.py
4. submit the eneb job calling run_eneb-vasp.py
   fe.out shows the residual force after each iteration
   structure of each image, ?.CON, is updated on the fly. To continue from an existing path, uncomment the lines in run_eneb_vasp.py that read in the path from *.CON and resubmit the script again.
5. check the MEP:
   tail -7 mep.out > neb.dat
   run nebspline.pl from VTSTSCRIPTS
   The above two lines will produce mep.eps as from a regular NEB run in VTSTcode


References:
  coming soon
