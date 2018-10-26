"""
Input file with all parameters and flags.
Import in python script with import params and call the parameters with params.x

NB: Coupling J is always set to 1.0
"""
L               = 8                 # SIZE
D               = 2.0               # DISORDER
Jz              = 1.0               # INTERACTION STRENGTH
BC              = 1                 # BOUNDARY CONDITIONS (0 ---> periodic, 1 ---> open)
Dis_gen         = 0                 # DISORDER FLAG (0 ---> random, 1 ---> quasiperiodic) ###
Format_flag     = 0                 # HAMILTONIAN FORMAT FLAG (0 ---> dense, 1 ---> sparse) ###
In_flag         = 1                 # INITIAL STATE FLAG (0 ---> random, 1 ---> 101010.., 2 ---> 1111100000)
t_i             = 0.0               # START TIME FOR EVOLUTION
steps           = 2500              # STEPS OF TIME EVOLUTION
N_real          = 500               # NUMBER OF RELIZATIONS
