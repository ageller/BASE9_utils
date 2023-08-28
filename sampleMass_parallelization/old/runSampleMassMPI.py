#!/software/anaconda3.6/bin/python


# MPI (mpi4py) Python script to execute all the commands in a file list

import argparse
import os
from mpi4py import MPI

def run_sampleMass_MPI(commandListFile, index):
	# read in the command list and execute the command at the given index

	with open(commandListFile) as f:
		lines = f.readlines()
		lines = [line.rstrip() for line in lines]

	cmd = lines[rank]
	print(index, cmd)
	os.system(cmd)


def define_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("-c", "--cname", 
		type=str, 
		help="output file name for command list [sampleMassCmds.sh]", 
		default='sampleMassCmds.sh'
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


	# get the rank, this will be the index
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()

	run_sampleMass_MPI(args.cname, rank)