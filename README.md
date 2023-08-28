# BASE9_utils
Code utilities for running and analyzing results from BASE-9 

Creating a conda env for this

```
conda create --name BASE9 -c conda-forge python=3.10 astropy astroquery jupyter scipy numpy matplotlib pandas pyyaml shapely bokeh
conda activate BASE9
```
# Introduction
This pipeline is for identifying photometric binaries in open clusters and works in conjunction with the Bayesian Analysis for Stellar Evolution with Nine Variables (BASE-9) code for open clusters in which Gaia, Pan-STARRS, and 2MASS photometry is available for.  More details on this pipeline may be found in Childs et al. (in prep).  Documentation for the BASE-9 code may be found at https://github.com/BayesianStellarEvolution/base-cpp.  All parts of the pipeline use dependencies from the virtual environment listed above.

# How to Use

  ## Generating BASE-9 Input Files

  The codes for generating the input files for the first stage of BASE-9 (the singlePop stage) are found in the getGaiaData folder.  The makePhot.ipynb notebook file, getGaiaData.py, and OCcompiled_clean_v2.csv files are needed.  The makePhot.ipynb notebook will interface with the getGaiaData.py code to query and format the observational data.  The output of these codes are the file.phot and file.yaml files needed for singlePop input.  These codes will also make use of Gaia kinematic measurements, where avaialable for stars, to determine the cluster membership prior (CMprior) of each star.
  
  In order to run the makePhot.ipynb notebook the user must have also downloaded the OCcompiled_clean_v2.csv file which contains the cluster prior values for the BASE-9 analysis.  More specifically, the file contains the center coordinates, cluster mass, age, distance, reddening, and metalliticity for a hundreds of Galactic open clusters.  These values are taken from previous literature and are necessary parameters for our pipeline.

  To run the makePhot.ipynb clusterName must be set to the name of cluster of interest, as it appears in the OCcompiled_clean_v2.csv file, in cell [2].  In cell [3] the user may apply differential reddening corrections by setting cluster.deredden = 1.  If the user does not wish to apply differential reddening corretions cluster.deredden should be set equal to 0.

  The user may also change the cluster.sig_fac number.  This parameter sets the number of sigmas for which we want to calculate CMpriors of the stars that fall within this sigma range.  The default for this value is set to 10 but should be reduced for a more selective sample of likely cluster members.

  cluster.runAll(clusterName) should be run if the user is querying the data for the first time.  If the user does not need to re-query the data and has a saved .csv of the data from a previous call of cluster.runAll(clusterName), the user should run cluster.runAll(clusterName,dataFile.csv) where dataFile.csv is the file with the previously queried data.  Running this code will produce the file.phot and file.yaml files as well as plots of the fits to the Gaia data in RV, parallax, propoer motion, and a CMD with cluster members shown in pink.

  In the last cell (cell [5]) there is an interactive isochrone tool.  This tool allows the user to adjust cluster priors and see how the prior values affect the isochrone and check its fit to the star in the file.phot file.  The filters shown on the CMD may be changed with the mag, color1, and color2 arguments.  The list of filters available are commented in cell [4].

  ## Adding Noise to simCluster data
