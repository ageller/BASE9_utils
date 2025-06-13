# BASE9_utils
Code utilities for running and analyzing results from BASE-9 

# Introduction
This pipeline is for identifying photometric binaries in open clusters and works in conjunction with the Bayesian Analysis for Stellar Evolution with Nine Variables (BASE-9) code for open clusters in which Gaia, Pan-STARRS, and 2MASS photometry is available for.  More details on this pipeline may be found in Childs et al. (in prep).  Documentation for the BASE-9 code may be found at https://github.com/BayesianStellarEvolution/base-cpp.  All parts of the pipeline use dependencies from the virtual environment, BASE9.  To create this conda environment run

```
conda create --name BASE9 -c conda-forge python=3.10 astropy astroquery jupyter scipy numpy matplotlib pandas pyyaml shapely bokeh=3.7.3 os hdbscan
conda activate BASE9
```

# How to Use
This pipeline is comprised of three major parts that must be run sequentially:
	1. Identifying potential OC members using HDBSCAN, generating the file.phot and file.yaml input files for BASE-9's singlePop.
 	2. Parallelizing the results from singlePop for a parallelzied version of sampleMass to produce the file.df file.
  	3.  Adding noise modeled on the observational noise in file.df to the simulated photometry from simCluster and then running this noisy simulated photometry through parallelized sampleMass to test for completeness.
  ## Generating BASE-9 input files for singlePop

  The codes for identifying possible OC members and generating the input files for the first stage of BASE-9 (the singlePop stage) are found in the getGaiaData folder.  The makePhot.ipynb notebook file, getGaiaData.py, template.yaml, PARSEC.model, and Hunt2023.tsv files are needed.  The makePhot.ipynb notebook will interface with the getGaiaData.py code to query and format the observational data.  The output of these codes are the file.phot and file.yaml files needed for singlePop input.  These codes will also make use of Gaia kinematic measurements, where avaialable for stars, to determine the cluster membership prior (CMprior) of each star.  More details for this portion of our pipeline are found in the getGaiaData folder.
  
  In order to run the makePhot.ipynb notebook the user must have also downloaded the Hunt2023.tsv file which contains the cluster prior values for the BASE-9 analysis.  More specifically, the file contains the center coordinates, cluster mass, age, distance, reddening, and metalliticity for a hundreds of Galactic open clusters.  These values are necessary parameters for our pipeline.

  cluster.run_pipeline(clusterName, query=True) should be run if the user is querying the data for the first time.  If the user does not need to re-query the data and has a saved .csv of the data from a previous run of the pipeline, the user should run cluster.run_pipeline(clusterName,query=False) and set the pathway for filename in the cluster.runAll(clusteName, filename) where the previously queried data is located.  Running this code will produce the file.phot and file.yaml files as well as plots of the fits to the Gaia data in RV, parallax, propoer motion, and a CMD with cluster members.

  If the user wishes to save the cluster object as a pickle object so it may be reloaded later, the following cells are for this option.  If the user has download the .ecsv file directly from the Gaia website, they should run the add_to_download(cluster_name) function to add additona columns to the downloaded file that are needed to run the pipeline.

  In the last cell there is an interactive isochrone tool.  This tool allows the user to adjust cluster values to see how it will affect the isochrone and check its fit to the star in the file.phot file.  These adjustments in cluster values will be applied to the starting values in the file.yaml.  The interactive tool also allows user to remove stars from file.phot that may through off the BASE-9 fit (i.e blue stragglers, possible field stars, etc.).  The filters shown on the CMD may be changed with the mag, color1, and color2 arguments.  The user may use any of the Gaia, 2MASS, or PS filters.  Once the user has found an isochrone, they may generate the file.phot and file.yaml files with the updated photometry and starting values in the file.yaml.

If the parallelized version of singlePop is being used (see BASE-9 docs for more details) and each chain is in a different directory named Run$i, where $i is an int 1-10, the chains may be combined back together with the combine_res.py code found in the getGaiaData folder.  The clusterName must be set in this code before running it.

## BEST PRACTICES FOR SINGLEPOP
To avoid logPost = -inf,make sure the phot file is properly formated. It is very particular. 
CMprior should be between 0.01 and 0.9 for singlePop.  A continuous logPost = -inf means there is a problem with either the phot file or the yaml file (make sure there are no nans for the prior, sigma, or starting values).  It is possible for a sampling to start in logPost = -inf and move out of it.
  ##  Parallelizing sampleMass

  After running singlePopMcmc, for the open cluster project, we want to run sampleMass.  This takes quite a long time, but can be split to run in parallel.  To do this, we need to split the phot file and run a separate instance of sampleMass on each subset of the phot file.  The dividePhot.py code in the sampleMassParllelization folder will:
	1. trim the .res file to include only stage 3 (trim_res)
	2. divide the phot file into chunks (divide_phot)
	3. generate a job script for Quest to run sampleMass in parallel

example command:
` python dividePhot.py --res ngc188.res --phot NGC_188.phot --yaml base9.yaml --nthreads 4 `

## NOTE: This will trim your res file to only include stage 3.  You should make a copy of it to keep the original version.

A typical number of threads to use for open clusters on Quest is 500.  After sampleMass is done running the files may be combined into one condensed file that summarizes the results with the sampleMassAnalysis.ipynb notebook.  To run the code in a Jupyter notebook call write_data(clusterName).  This will read in each partion of the parallelized sampleMass output and summarize the results in a file.df file.  The file.df file will contain the columns:  

` source_id ra dec pmra pmdec radial_velocity G phot_g_mean_flux_over_error G_BP phot_bp_mean_flux_over_error G_RP phot_rp_mean_flux_over_error parallax teff_gspphot ruwe number_of_neighbours number_of_mates g_ps sigg_ps r_ps sigr_ps i_ps sigi_ps z_ps sigz_ps y_ps sigy_ps J_2M sigJ_2M H_2M sigH_2M Ks_2M sigKs_2M rCenter PPa sig_E(B-V) E(B-V) PRV PM_ra PM_dec PPM CMprior member binary m1Median qMedian m1Std m1_16 m1_84 qStd q_16 q_84 `

and each row will contain these data for every star sent through sampleMass.  `source_id` is the Gaia source ID of the star, `ra` and `dec` are the right ascension and declination of the star in degrees, `pmra` and `pmdec` are the proper motion measurements of the star in right ascension and dclination in mas/yr, `radial_velocity` is the RV measurement of the star in km/s.  `G`, `phot_g_mean_flux_over_error`, `G_BP`, `phot_bp_mean_flux_over_error`, `G_RP`, `phot_rp_mean_flux_over_error` are the Gaia photometric measurements and errors taken in Gaia G, G_BP, and G_RP bandpasses. `parallax` is the parallax measurement of the star in mas.  `teff_gspphot` is the effective temperature estimate and `ruwe` is the ruwe value from Gaia. The `number_of_neighbours` and `number_of_mates` are ancillary data from Gaia catalog matching. `g_ps`, `sigg_ps`, `r_ps`, `sigr_ps`, `i_ps`, `sigi_ps`, `z_ps`, `sigz_ps`, `y_ps`, and `sigy_ps` are the photometric measurments and errors for the Pan-STARRS g, r, i, z, and y bandpasses, respectively.  `J_2M`, `sigJ_2M`, `H_2M`, `sigH_2M`, `Ks_2M`, `sigKs_2M`, are the photometric measurements and errors for the 2MASS survery in the J, H, and Ks bandpasses, respectively. `rCenter` is the angular distance from the star from the cluster center in degrees. `PPa` is the probability the star is a cluster member as determined by its parallax measurment, `sig_E(B-V)` and `E(B-V)` are the error and reddening value taken from the Bayestarr reddening map. `PRV` is the probability the star is a cluster member as determined by its radial velocity measurment, `PM_ra`, `PM_dec` and `PPM` are the probabilities the star is a cluster member as determined by its proper motion in the right ascension, propoer motion in declination and combined probability for proper motion measuremnts.  The `CMprior` is the final cluster membership probability we feed to BASE-9.  The following collumns contain information from the output of BASE-9 singlePop: `member` and `binary` are True or False if BASE-9 determined the star was either a member and/or binary. `m1Median` is the median of the posterior distribution for primary mass and `qMedian` is the median of the posterior distribution for mass ratio.  `m1Std`, `m1_16`, and  `m1_84` are the standard deviation, 16th and 84th percentiles of the posterior distribution for primary mass, and `qStd`, `q_16` and `q_84` are the standard deviation, 16th and 84th percentiles of the posterior distribution for the mass ratio.  All of the kinematic measurements have been taken by Gaia.

  ## Adding noise to simCluster data
  To test for completeness, synthetic noise modeled after the noise in the observational data may be added to the simulated data from simCluster.  To do so requires the file.df file from the sampleMassAnalysis.ipynb code, the simulated phot file (file.sim.out), and the add_sim_noise.py code which can be found in the simCluster_utils folder.  To code requires two flags to be set.  The -n flag requires the name of the cluster and the -m flag requires the minimum Gaia G magnitude for which to apply noise to stars with G magnitudes in between 20 and -m.  For example, to run the code on NGC 188 for stars dimmer than G=12 the command is
  ` python3 add_sim_noise.py -n NGC_188 -m 12 `

  This will produce the output file simCluster.phot which has noise added to simulated data for stars in the G magnitude range 12 to 20.  This simCluster.phot may then be ran through the parallelized version of sampleMass in the same way as described above and the results can be used to test for completeness.
