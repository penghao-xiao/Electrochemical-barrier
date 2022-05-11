# Electrochemical-barrier
This package performs structure optimization and saddle search under constant potential. The geometry and number of electrons are optimized simultaneously without adding significant overhead compared to constant-charge calculations. The current implementation is based on ASE and VASPsol. 

Install:
 1. Make sure VASPsol is compided
 2. pip3 install --upgrade --user ase  
 3. Set ASE with VASPsol as the binary, see script/run_vasp.py for the srun/mpirun setting
    (potcar files and others, see https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#environment-variables)
 4. In bash_profile: export PYTHONPATH=$PYTHONPATH:your_path/Electrochemical-barrier/

Usage:
1. Before doing any calculation, first set the vacuum/solution along the z axis and move the slab to the center of the simulation box using scripts/center_slab.py (do not run the script separately for the initial and final states because there could be some small mismatch for the substrate.)
2. Optimize the end points with optimization/constV_opt.py
   - grep FIRE slurm... to check the magnitude of residual forces (force, mu_e)
   - grep electron slurm... for number of electrons and mu_e
3. Copy the initial and final structures to 0.CON and 6.CON for an eNEB run
4. Set the optimized number of electrons for the initial and fianl states in ne1=... and ne2=... in example/run_eneb-vasp.py
5. Submit the slurm job to run run_eneb-vasp.py
   - fe.out shows the residual force after each iteration
   - structure of each image, ?.CON, is updated on the fly. To continue from an existing path, uncomment the lines in run_eneb_vasp.py that read in the path from ?.CON and resubmit the script again.
6. Check the MEP:
   - tail -7 mep.out > neb.dat
   - nebspline.pl (from the VTSTSCRIPTS)
   - The above two commands will produce mep.eps as from a regular NEB run in VTSTcode


References:
  Z. Duan and P. Xiao, Simulation of Potential-Dependent Activation Energies in Electrocatalysis: Mechanism of O−O Bond Formation on RuO2, *J. Phys. Chem. C* 2021, 125, 28, 15243
