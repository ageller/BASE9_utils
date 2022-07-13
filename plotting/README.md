## From Elizabeth Jeffery
- **mcmc_results.py**: shows on the screen the average and standard deviation of each of the four parameters (post burn-in) in the .res file. It runs on the command line; usage: `./mcmc_results.py mystar.res`

- **plot_mcmc.py**: makes two plots from the .res file; one is the trace plot (post burn-in) for each of the four parameters, the other is a histogram of the mcmc chain. Usage: `./plot_mcmc.py mystar.res`

- **stats_sampleWDMass.py**: prints to the screen the average and standard deviation (post burn-in) of each parameter sampled on by sampleWDMass. Usage: `./stats_sampleWDMass.py mystar.res`

- **check_cmd.py**: make a quick CMD of a .phot file (including error bars), outputs cmd.png.  Usage: `./check_cmd.py <phot file> <y-axis filter> <x-axis color>` e.g., `./check_cmd.py hyades.phot V B-V`

As can be seen on the first line of each script, it has the python path set as `#!/usr/bin/python3.8`. This may need to be changed, depending on where the python executables are stored. It also assumes you have numpy as matplotlib installed, which most python users do.

Note: you can also run these scripts as regular .py files (not executables), with, e.g.,  `python mcmc_results.py mystar.res`.  In that case, the first line of the script (with the path to python) is not relevant.  
