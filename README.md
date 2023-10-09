# BASE-9 utils
Code utilities for running and analyzing results from BASE-9 

# Introduction
The codes in this repo provide a pipeline for identifying photometric binaries in open clusters and works in conjunction with the Bayesian Analysis for Stellar Evolution with Nine Variables (BASE-9) code.  Here we download  Gaia, Pan-STARRS, and 2MASS photometry and create input files for BASE-9.  We also provide various analysis and validation codes for BASE-9 results.  More details on this pipeline can be found in Childs et al. (in prep).  Documentation for the BASE-9 code can be found at https://github.com/BayesianStellarEvolution/base-cpp.  All parts of the pipeline use dependencies that can be included in a the virtual environment, BASE9, using `conda` by running the following commands in a terminal:

```
conda create --name BASE9 -c conda-forge python=3.10 astropy astroquery jupyter scipy numpy matplotlib pandas pyyaml shapely bokeh
conda activate BASE9
pip install dustmaps
```

*Note*: you also need to have a working version of latex installed in order to produce the figures created with these codes.  
# How to Use
This pipeline is comprised of three major parts that must be run sequentially:

1. Generating the `file.phot` and `file.yaml` input files for BASE-9's `singlePopMcmc`
2. Formatting the results from `singlePopMcmc` for a parallelized version of BASE-9's `sampleMass` to produce the `file.df` file.
3. Adding noise modeled on the observational noise in `file.df` to the simulated photometry from BASE-9's `simCluster` and then running this noisy simulated photometry through parallelized `sampleMass` to test for completeness.

## 1. Generating BASE-9 input files for `singlePopMcmc`

The codes for generating the input files for the first stage of BASE-9 (the `singlePopMcmc` stage) are found in the `getGaiaData` folder.  The `makePhot.ipynb` notebook file, `getGaiaData.py`, and `OCcompiled_clean_v2.csv` files are needed.  The `makePhot.ipynb` notebook will interface with the `getGaiaData.py` code to query and format the observational data.  The output of these codes are the `file.phot` and `file.yaml` files (whose names can be defined in the code) needed for `singlePopMcmc` input.  These codes will also make use of Gaia kinematic and distnace measurements, where avaialable for stars, to determine the cluster membership prior (`CMprior`) of each star.

In order to run the `makePhot.ipynb` notebook, the user must have also downloaded the `OCcompiled_clean_v2.csv` file which contains the cluster prior values for the BASE-9 analysis.  More specifically, the file contains the center coordinates, cluster mass, age, distance, reddening, and metalliticity for a hundreds of Galactic open clusters.  These values are taken from previous literature, and compiled in [this GitHub repo](https://github.com/ageller/compileOCs) and are necessary parameters for our pipeline.  

To run the `makePhot.ipynb`, the `clusterName` variable must be set to the name of cluster of interest, as it appears in the `OCcompiled_clean_v2.csv` file.  The user can apply differential reddening corrections by setting `cluster.deredden = True`.  If the user does not wish to apply differential reddening corretions set `cluster.deredden = False`.

The user may also change the `cluster.sig_fac` number, which sets the number of standard deviations for which we want to calculate `CMpriors` of the stars.  The default for this value is set to 10 but should be reduced for a more selective sample of likely cluster members.  (Stars ourside of `cluster.sig_fac` are excluded as field stars.)

`cluster.runAll(clusterName)` should be run if the user is querying the data for the first time.  If the user does not need to re-query the data and has a saved a `.ecsv` of the data from a previous call of `cluster.runAll(clusterName)`, the user should run `cluster.runAll(clusterName,dataFile.ecsv)`, where `dataFile.ecsv` is the file with the previously queried data.  Running this code will produce the `file.phot` and `file.yaml` files as well as plots of the fits to the Gaia data in RV, parallax, proper motion, and a CMD with cluster members shown in pink.

In the last cell there is an interactive isochrone tool.  This tool allows the user to adjust cluster values to see how it will affect the isochrone and check its fit to the star in the `file.phot` file.  (You will also need to download the `PARSEC.model` file from the [BASE-9 models GitHub repo](https://github.com/BayesianStellarEvolution/base-models) in order to run this, as also stated in the notebook.)  These adjustments in cluster values will be applied to the starting values in the `file.yaml`.  The interactive tool also allows user to remove additional field stars from `file.phot` and also cluster members that may throw off the BASE-9 fit (e.g., blue stragglers, etc.).  The filters shown on the CMD can be changed with the `mag`, `color1`, and `color2` arguments.  The list of filters available are commented in the notebook.  Once the user has found an isochrone, they can generate the `file.phot` and `file.yaml` files with the updated photometry and starting values.

There is another interactive tool, though not shown in the example .ipynb file: `cluster.createInteractiveSelector`, which can be used to interactively select regions in kinematic and distance space that may define the cluster and see those selected stars on a CMD.  This may be useful for a cluster that is heavily embedded in a rich field where the automated algorithms in our code have difficulty selecting members.  Using `cluster.createInteractiveSelector`, the user can identify helpful limits to trim down the data before sending it through `cluster.runAll`.  (Note: this interactive is also built in `Bokeh` and should be executed in a similar manner as `cluster.createInteractiveIsochrone`.)

If the parallelized version of `singlePopMcmc` is being used (see BASE-9 docs for more details) and each chain is in a different directory named `Run$i`, where `$i` is an int 1-10, the chains can be combined back together with the `combine_res.py` code found in the `getGaiaData` folder.  The clusterName must be set in this code before running it.

We provide codes within the `plotting/` directory of this repo that can be used to generate summary plots and statistics for the posterior distributions resulting from `singlePopMcmc`.  Please see the [README](https://github.com/ageller/BASE9_utils/blob/main/plotting/README.md) in that directory for more information.

##  2. Parallelizing `sampleMass`

After running `singlePopMcmc`, you can run `sampleMass` to derive star-by-star masses, mass-ratios and photometric membership probabilities.  This takes quite a long time but can be split to run in parallel.  To do this, we need to split the `file.phot` file and run a separate instance of `sampleMass` on each subset of the `file.phot` file.  The `dividePhot.py` code in the `sampleMass_parallelization` directory will:
1. trim the `.res` file to include only stage 3 (`trim_res`)
2. divide the `.phot` file into chunks (`divide_phot`)
3. generate a job script for [Quest](https://www.it.northwestern.edu/departments/it-services-support/research/computing/quest/) (using Slurm) to run `sampleMass` in parallel

example command:
```
python dividePhot.py --res ngc188.res --phot NGC_188.phot --yaml base9.yaml --nthreads 4
```

**NOTE: This will trim your res file to only include stage 3.  You should make a copy of it to keep the original version.**

A typical number of threads to use for open clusters on Quest is 500.  After `sampleMass` is done running, the resulting files can be combined into one condensed file that summarizes the results with the `sampleMassParallelization/sampleMassAnalysis.ipynb` notebook.  To run the code in a Jupyter notebook call `write_data(clusterName)`.  This will read in each partion of the parallelized `sampleMass` output and summarize the results in a `file.df` file.  The `file.df` file contain for every star sent through `sampleMass` (in separate rows) with the columns:  

- `source_id` : Gaia source ID of the star
- `ra`, `dec` : the right ascension and declination of the star in degrees
- `pmra`, `pmdec` : the proper motion measurements of the star in right ascension and dclination in mas/yr
- `radial_velocity` : the RV measurement of the star in km/s
- `G`, `phot_g_mean_flux_over_error`, `G_BP`, `phot_bp_mean_flux_over_error`, `G_RP`, `phot_rp_mean_flux_over_error` : the Gaia photometric measurements and errors taken in Gaia G, G_BP, and G_RP bandpasses
- `parallax` : the parallax measurement of the star in mas 
- `teff_gspphot` :  the effective temperature estimate from Gaia 
- `ruwe` : the ruwe value from Gaia
- `number_of_neighbours`, `number_of_mates` : ancillary data from Gaia catalog matching 
- `g_ps`, `sigg_ps`, `r_ps`, `sigr_ps`, `i_ps`, `sigi_ps`, `z_ps`, `sigz_ps`, `y_ps`,  `sigy_ps` : the photometric measurments and errors for the Pan-STARRS g, r, i, z, and y bandpasses, respectively
- `J_2M`, `sigJ_2M`, `H_2M`, `sigH_2M`, `Ks_2M`, `sigKs_2M` : the photometric measurements and errors for the 2MASS survery in the J, H, and Ks bandpasses, respectively
- `rCenter` : the angular distance from the star from the cluster center in degrees
- `sig_E(B-V)`,  `E(B-V)` : the error and reddening value taken from the Bayestarr reddening map. 
- `PPa` : the probability the star is a cluster member as determined by its Gaia parallax measurment calaculated within our code
- `PRV` : the probability the star is a cluster member as determined by its Gaia radial-velocity measurement calculated within our code
-  `PM_ra`, `PM_dec`, `PPM` :  the probabilities the star is a cluster member as determined by its Gaia proper motion in the right ascension, proper motion in declination and combined probability for proper motion measuremnts calculated within our code
- `CMprior` : the final cluster membership probability we feed to BASE-9 (calculated from our code)
- `member`, `binary` : (True/False) our final identification of cluster members and binaries, resulting from our analysis of BASE-9 results  
- `m1Median`, `m1Std`, `m1_16`, `m1_84` : the median, standard deviation, 16th and 84th percentiles of the BASE-9 posterior distribution for primary mass 
- `qMedian`,`qStd`, `q_16`, `q_84` : the median, standard deviation, 16th and 84th percentiles of the BASE-9 posterior distribution for mass ratio.





## 3. Adding noise to `simCluster` data
To test for completeness, synthetic noise modeled after the noise in the observational data can be added to the simulated data from `simCluster`.  To do so requires the `file.df` file from the `sampleMassAnalysis.ipynb` code, the simulated phot file (`file.sim.out`), and the `add_sim_noise.py` code which can be found in the `simCluster_utils` folder.  The code requires two flags to be set.  The `-n` flag requires the name of the cluster, and the `-m` flag requires the minimum Gaia *G* magnitude for which to apply noise to stars with *G* magnitudes in between 20 and `-m`.  For example, to run the code on NGC 188 for stars dimmer than *G*=12 the command is

```
python3 add_sim_noise.py -n NGC_188 -m 12
```

This will produce the output file `simCluster.phot` which has noise added to simulated data for stars in the *G* magnitude range 12 to 20.  This `simCluster.phot` can then be run through the parallelized version of `sampleMass` in the same way as described above, and the results can be used to test for completeness.
