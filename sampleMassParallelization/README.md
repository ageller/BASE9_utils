After running singlePopMcmc, for the open cluster project, we want to run sampleMass.  This takes quite a long time, but can be split to run in parallel.  To do this, we need to split the phot file and run a separate instance of sampleMass on each subset of the phot file.  The code in here is my first attempt at doing this.

dividePhot.py
Python script to :
	1. trim the .res file to include only stage 3 (trim_res)
	2. divide the phot file into chunks (divide_phot)
	3. generate a job script for Quest to run sampleMass in parallel

example command:
` python dividePhot.py --res ngc188.res --phot NGC_188.phot --yaml base9.yaml --nthreads 4 `
