'''
Python script to :
	1. trim the .res file to include only stage 3 (trim_res)
	2. divide the phot file into chunks (divide_phot)
	3. generate a job script for Quest to run sampleMass in parallel

example command:
	python dividePhot.py --res ngc188.res --phot NGC_188.phot --yaml base9.yaml --nthreads 4
'''


import pandas as pd
import numpy as np
import argparse
import shutil
import subprocess, sys, shlex

def trim_res(resFile):

	# save the original res file
	shutil.copy(resFile, resFile + '.org')

	# trim the file using awk
	cmd = f"cat {resFile}.org | awk '{{if (NR == 1 || $6 == 3) print $0}}' > {resFile}"
	process = subprocess.Popen(shlex.split(cmd), 
		stdout=subprocess.PIPE, 
		stderr=subprocess.PIPE, 
		shell=True)

	stdout, stderr = process.communicate()

def divide_phot(resFile, photFile, yamlFile, nThreads):

	# read in the phot file (as strings so that I keep the formatting)
	df = pd.read_csv(photFile, sep="\s+", converters={i: str for i in range(100)})

	# generate nthreads subsets of the res file
	split_df = np.array_split(df, nThreads)
	#print(split_df)

	# for each subset
	#   save the subset of the phot file 
	#   copy the res file so that it has the same root file name as the phot file

	# get the zfill amount
	nfill = int(np.ceil(np.log10(nThreads)))
	# get the root file name from the phot file
	fnameRoot = photFile.replace('.phot','')

	for i,usedf in enumerate(split_df):
		destRoot = fnameRoot + '_' + str(i).zfill(nfill)

		# save that phot file
		destPhot = destRoot + '.phot'
		usedf.to_csv(destPhot, index=None, sep=' ')

		# copy the res file
		shutil.copy(resFile, destRoot + '.res')

		# create the command
		#cmd = f"sampleMass --config {yamlFile} --photFile {destPhot} --outputFileBase {destRoot}"
		#print(cmd)



def create_srun(nThreads, srunName, yamlFile, photFile, srunFile):
	fnameRoot = photFile.replace('.phot','')

	# create an jobarray script that can be used on Quest
	cmd = ""
	cmd += "#!/bin/bash\n"
	cmd += "#SBATCH --account=p31721\n"
	cmd += "#SBATCH --partition=long\n"
	cmd += f"#SBATCH --array=0-{nThreads - 1}\n"
	cmd += "#SBATCH --nodes=1\n"
	cmd += "#SBATCH --ntasks-per-node=1\n"
	cmd += "#SBATCH --time=168:00:00\n"
	cmd += "#SBATCH --mem-per-cpu=1G\n"
	cmd += f"#SBATCH --job-name=\"{srunName}_${{SLURM_ARRAY_TASK_ID}}\"\n" 
	cmd += "#SBATCH --output=\"jobout.%A_%a\"\n"
	cmd += "#SBATCH --error=\"joberr.%A_%a\"\n\n"
	cmd += "printf \"Deploying job ...\"\n"
	cmd += "scontrol show hostnames $SLURM_JOB_NODELIST\n"
	cmd += "echo $SLURM_SUBMIT_DIR\n"
	cmd += "printf \"\\n\"\n\n"
	cmd += "export PATH=$PATH:/projects/p31721/BASE9/bin\n"
	cmd += "module purge all\n\n"
	# cmd += "module load gcc/4.8.3\n"
	# cmd += "module load gsl/2.1-gcc4.8.3\n"
	# cmd += "module load cmake/3.15.4\n" #possibly not needed
	# cmd += "module load boost/1.70.0\n"
	cmd += f"sampleMass --config {yamlFile} --photFile {fnameRoot}_${{SLURM_ARRAY_TASK_ID}}.phot --outputFileBase {fnameRoot}_${{SLURM_ARRAY_TASK_ID}}\n\n"
	cmd += "printf \"==================== done with sampleMass ====================\"\n"
	
	with open(srunFile, 'w',  newline='\n') as f:
		f.write(cmd)


def define_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("-r", "--res",
		type=str,
		help=".res file name [base9.res]", 
		default="base9.res"
	)
	parser.add_argument("-p", "--phot", 
		type=str, 
		help=".phot file name [base9.phot]", 
		default="base9.phot"
	)
	parser.add_argument("-y", "--yaml", 
		type=str, 
		help=".yaml file name [base9.yaml]", 
		default="base9.yaml"
	)
	parser.add_argument("-n", "--nthreads", 
		type=int, 
		help="number of threads [10]", 
		default=10
	)
	parser.add_argument("-a", "--srunName", 
		type=str, 
		help="name for the Quest process [BASE9]", 
		default='BASE9'
	)
	parser.add_argument("-s", "--srunFile", 
		type=str, 
		help="name for the srun file [srun_array.sh]", 
		default='srun_array.sh'
	)

	#https://docs.python.org/2/howto/argparse.html
	args = parser.parse_args()
	#to print out the options that were selected (probably some way to use this to quickly assign args)
	opts = vars(args)
	options = { k : opts[k] for k in opts if opts[k] is not None }
	print(options)

	return args

if __name__ == "__main__":

	args = define_args()
	trim_res(args.res)
	divide_phot(args.res, args.phot, args.yaml, args.nthreads)
	create_srun(args.nthreads, args.srunName, args.yaml, args.phot, args.srunFile)

