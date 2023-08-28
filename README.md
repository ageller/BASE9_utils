# BASE9_utils
Code utilities for running and analyzing results from BASE-9 

# Introduction
This pipeline is for identifying photometric binaries in open clusters and works in conjunction with the Bayesian Analysis for Stellar Evolution with Nine Variables (BASE-9) code for open clusters in which Gaia, Pan-STARRS, and 2MASS photometry is available for.  More details on this pipeline may be found in Childs et al. (in prep).  Documentation for the BASE-9 code may be found at https://github.com/BayesianStellarEvolution/base-cpp.  All parts of the pipeline use dependencies from the virtual environment, BASE9.  To create this conda environment run

```
conda create --name BASE9 -c conda-forge python=3.10 astropy astroquery jupyter scipy numpy matplotlib pandas pyyaml shapely bokeh
conda activate BASE9
```

# How to Use

  ## Generating BASE-9 input files

  The codes for generating the input files for the first stage of BASE-9 (the singlePop stage) are found in the getGaiaData folder.  The makePhot.ipynb notebook file, getGaiaData.py, and OCcompiled_clean_v2.csv files are needed.  The makePhot.ipynb notebook will interface with the getGaiaData.py code to query and format the observational data.  The output of these codes are the file.phot and file.yaml files needed for singlePop input.  These codes will also make use of Gaia kinematic measurements, where avaialable for stars, to determine the cluster membership prior (CMprior) of each star.
  
  In order to run the makePhot.ipynb notebook the user must have also downloaded the OCcompiled_clean_v2.csv file which contains the cluster prior values for the BASE-9 analysis.  More specifically, the file contains the center coordinates, cluster mass, age, distance, reddening, and metalliticity for a hundreds of Galactic open clusters.  These values are taken from previous literature and are necessary parameters for our pipeline.

  To run the makePhot.ipynb clusterName must be set to the name of cluster of interest, as it appears in the OCcompiled_clean_v2.csv file, in cell [2].  In cell [3] the user may apply differential reddening corrections by setting cluster.deredden = 1.  If the user does not wish to apply differential reddening corretions cluster.deredden should be set equal to 0.

  The user may also change the cluster.sig_fac number.  This parameter sets the number of sigmas for which we want to calculate CMpriors of the stars that fall within this sigma range.  The default for this value is set to 10 but should be reduced for a more selective sample of likely cluster members.

  cluster.runAll(clusterName) should be run if the user is querying the data for the first time.  If the user does not need to re-query the data and has a saved .csv of the data from a previous call of cluster.runAll(clusterName), the user should run cluster.runAll(clusterName,dataFile.csv) where dataFile.csv is the file with the previously queried data.  Running this code will produce the file.phot and file.yaml files as well as plots of the fits to the Gaia data in RV, parallax, propoer motion, and a CMD with cluster members shown in pink.

  In the last cell (cell [5]) there is an interactive isochrone tool.  This tool allows the user to adjust cluster priors and see how the prior values affect the isochrone and check its fit to the star in the file.phot file.  The filters shown on the CMD may be changed with the mag, color1, and color2 arguments.  The list of filters available are commented in cell [4].

  ##  Parallelizing sampleMass

  After running singlePopMcmc, for the open cluster project, we want to run sampleMass.  This takes quite a long time, but can be split to run in parallel.  To do this, we need to split the phot file and run a separate instance of sampleMass on each subset of the phot file.  The dividePhot.py code in the sampleMassParllelization folder will:
	1. trim the .res file to include only stage 3 (trim_res)
	2. divide the phot file into chunks (divide_phot)
	3. generate a job script for Quest to run sampleMass in parallel

example command:
` python dividePhot.py --res ngc188.res --phot NGC_188.phot --yaml base9.yaml --nthreads 4 `

## NOTE: This will trim your res file to only include stage 3.  You should make a copy of it to keep the original version.

A typical number of threads to use for open clusters on Quest is 500.  After sampleMass is done running the files may be combined into one condensed file that summarizes the results with the sampleMassAnalysis.ipynb notebook.  To run the code in a Jupyter notebook call write_data(clusterName).  This will read in each partion of the parallelized sampleMass output and summarize the results in a file.df file.  The file.df file will contain the columns:  

` source_id ra dec pmra pmdec radial_velocity G phot_g_mean_flux_over_error G_BP phot_bp_mean_flux_over_error G_RP phot_rp_mean_flux_over_error parallax teff_gspphot ruwe number_of_neighbours number_of_mates g_ps sigg_ps r_ps sigr_ps i_ps sigi_ps z_ps sigz_ps y_ps sigy_ps J_2M sigJ_2M H_2M sigH_2M Ks_2M sigKs_2M sigG sigG_BP sigG_RP coord.ra coord.dec rCenter id PPa sig_E(B-V) E(B-V) PRV PM_ra PM_dec PPM CMprior member binary m1Median qMedian m1Std m1_16 m1_84 qStd q_16 q_84 `

and each row will contain these data for every star sent through sampleMass.


  ## Adding noise to simCluster data
  To test for completeness, synthetic noise modeled after the noise in the observational data may be added to the simulated data from simCluster.  To do so requires the file.df file from the sampleMassAnalysis.ipynb code, the simulated phot file (file.sim.out), and the add_sim_noise.py code which can be found in the simCluster_utils folder.  To code requires two flags to be set.  The -n flag requires the name of the cluster and the -m flag requires the minimum Gaia G magnitude for which to apply noise to stars with G magnitudes in between 20 and -m.  For example, to run the code on NGC 188 for stars dimmer than G=12 the command is
  ` python3 add_sim_noise.py -n NGC_188 -m 12 `

  This will produce the output file simCluster.phot which has noise added to simulated data for stars in the G magnitude range 12 to 20.  This simCluster.phot may then be ran through the parallelized version of sampleMass in the same way as described above and the results can be used to test for completeness.
