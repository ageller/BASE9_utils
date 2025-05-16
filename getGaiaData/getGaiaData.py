# Python code to download Gaia data at a given location

from astroquery.gaia import Gaia
from astropy.modeling.models import KingProjectedAnalytic1D
from astropy.modeling import models, fitting
from astropy.table import Table, Column, vstack, MaskedColumn
import astropy.units as units
from astropy.coordinates import SkyCoord
#from dustmaps.bayestar import BayestarWebQuery
from astropy.io import ascii
import scipy.stats
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mplColors
import matplotlib.cm as cm
import yaml
from bokeh import *
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Button, PointDrawTool, Slider, TextInput, Div, Paragraph
from bokeh.layouts import column, row
from bokeh.io import curdoc
from bokeh.transform import factor_cmap
from shapely.geometry import LineString as shLs
from shapely.geometry import Point as shPt
from scipy.interpolate import interp1d, RegularGridInterpolator
import pandas as pd
from astropy.table import Table, join, hstack
from scipy.integrate import quad
from matplotlib.patches import Polygon
from scipy import stats
import hdbscan
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
def gauss(x, A, mu, sigma):
	return A*np.exp(-(x-mu)**2/(2*sigma**2))
def bi_norm(x, *args):
	A1, m1, s1, A2, m2,s2 = args
	#ret = A1*scipy.stats.norm.pdf(x, loc=m1 ,scale=s1) + A2*scipy.stats.norm.pdf(x, loc=m2 ,scale=s2)
	ret = gauss(x,A1,m1,s1)+gauss(x,A2,m2,s2)
	return ret
def tri_norm(x, *args):
	A1, m1, s1, A2, m2,s2, A3, m3,s3 = args
	ret = A1*scipy.stats.norm.pdf(x, loc=m1 ,scale=s1)
	ret += A2*scipy.stats.norm.pdf(x, loc=m2 ,scale=s2)
	ret += A3*scipy.stats.norm.pdf(x, loc=m3 ,scale=s3)
	return ret
def quad_norm(x, *args):
	m1, s1, A1, m2,s2, A2, m3,s3, A3, m4,s4, A4 = args
	ret = A1*scipy.stats.norm.pdf(x, loc=m1 ,scale=s1)
	ret += A2*scipy.stats.norm.pdf(x, loc=m2 ,scale=s2)
	ret += A3*scipy.stats.norm.pdf(x, loc=m3 ,scale=s3)
	ret += A4*scipy.stats.norm.pdf(x, loc=m4 ,scale=s4)
	return ret

class GaiaClusterMembers(object):
	'''
	This Class will grab data from the Gaia archive.  The user must provide the RA and Dec values,
	and the Class will return the full catalog and 

	RA, Dec and radius should be provided in decimal degrees.
	
	'''
	
	def __init__(self, RA = None, Dec = None, radius = 1, **kwargs):
		
		# required inputs
		self.RA = RA # degrees
		self.Dec = Dec # degrees

		# radius of the area to return
		self.radius = radius # degrees

		# catalogs (when DR3 comes out, we will change that to the default)
		# self.catalog = "gaiadr2.gaia_source" 
		self.GaiaCatalog = "gaiadr3.gaia_source" 
		self.PanSTARRSMatchCatalog = "gaiadr3.panstarrs1_best_neighbour"
		self.PanSTARRSCatalog = "gaiadr2.panstarrs1_original_valid"
		self.TMASSMatchCatalog = "gaiadr3.tmass_psc_xsc_best_neighbour"
		self.TMASSJoinCatalog = "gaiadr3.tmass_psc_xsc_join"
		self.TMASSCatalog = "gaiadr1.tmass_original_valid"

		self.yamlTemplateFileName = "template_base9.yaml" #default demplate for the yaml file

		# maximum error that we will allow in a source to be retrieved (not sure what the best value is here)
		self.maxPMerror = 1 # mas/year

		# set to 1 or 2 to print out more (and even more) information
		self.verbose = 0
		


		# columns to select
		self.columns = ['gaia.source_id',
			'gaia.ra',
			'gaia.dec',
			'gaia.pmra',
			'gaia.pmdec',
			'gaia.radial_velocity',
			'gaia.phot_g_mean_mag',
			'gaia.phot_g_mean_flux_over_error',
			'gaia.phot_bp_mean_mag',
			'gaia.phot_bp_mean_flux_over_error',
			'gaia.phot_rp_mean_mag',
			'gaia.phot_rp_mean_flux_over_error',
			'gaia.parallax',
			'gaia.parallax_error',
			'gaia.teff_gspphot',
			'gaia.ruwe',
			'best.number_of_neighbours',
			'best.number_of_mates',
			'ps.g_mean_psf_mag',
			'ps.g_mean_psf_mag_error',
			'ps.r_mean_psf_mag',
			'ps.r_mean_psf_mag_error',
			'ps.i_mean_psf_mag',
			'ps.i_mean_psf_mag_error',
			'ps.z_mean_psf_mag',
			'ps.z_mean_psf_mag_error',
			'ps.y_mean_psf_mag',
			'ps.y_mean_psf_mag_error',
			'tmass.j_m',
			'tmass.j_msigcom',
			'tmass.h_m',
			'tmass.h_msigcom',
			'tmass.ks_m',
			'tmass.ks_msigcom',
		]


		self.deredden = 1
		self.red = False
		# initial guesses for membership
		self.RVmin = -100. #km/s
		self.RVmax = 100. #km/s
		self.RVbins = 50
		self.dmin = 0. #parsecs
		self.dmax = 7000. #parsecs
		self.dbins = 200
		self.dPolyD = 6 #degrees for polynomial fit for distance distribution
		self.PMxmin = -100 #mas/yr
		self.PMxmax = 100 #mas/yr
		self.PMxbins = 200
		self.PMymin = -100 #mas/yr
		self.PMymax = 100 #mas/yr
		self.PMybins = 200  
		self.RVmean = None #could explicitly set the mean cluster RV for the initial guess
		self.RVsigma = None
		self.distance = None #could explicitly set the mean cluster distance for the initial guess
		self.PMmean = [None, None] #could explicitly set the mean cluster PM for the initial guess
		self.r_tide = 0
		self.r_core = 0
		self.fitter = fitting.LevMarLSQFitter()
		self.mem_min = 1e-10


		self.photSigFloor = 0.01 # floor to the photometry errors for the .phot file
		#self.sig_fac = 3
		# output
		self.SQLcmd = ''
		self.data = None # will be an astropy table
		self.small_data = None
		self.group_no = None
		self.lim_radius = None
		self.min_cluster_size = None
		self.createPlots = True # set to True to generate plots
		self.plotNameRoot = ''
		self.photOutputFileName = 'input.phot'
		self.yamlOutputFileName = 'base9.yaml'


		# dict for yaml
		# lists give [start, prior mean, prior sigma]
		self.yamlInputDict = {
			'photFile' : self.photOutputFileName,
			'outputFileBase' : None,
			'modelDirectory' : '/projects/p31721/BASE9/base-models/',
			'msRgbModel' : 5,
			'Fe_H' : [0., 0.3, 0.3],
			'Av' : [0., 0.3, 0.3],
			'logAge' : [9., np.inf, np.inf],
			'distMod' : [10., 1., 1.],
			'Y' : [0.29, 0.0, 0.0],
			'carbonicity' : [0.38, 0.0, 0.0]
		}

	# to rename the model, given Gaia phot columns
		# for PARSEC models
		self.magRenamer = {
			'G':'phot_g_mean_mag',
			'G_BP' :'phot_bp_mean_mag',
			'G_RP':'phot_rp_mean_mag',
			'g_ps':'g_mean_psf_mag',
			'r_ps':'r_mean_psf_mag',
			'i_ps':'i_mean_psf_mag',
			'z_ps':'z_mean_psf_mag',
			'y_ps':'y_mean_psf_mag',
			'sigG':'phot_g_mean_mag_error',
			'sigG_BP' :'phot_bp_mean_mag_error',
			'sigG_RP':'phot_rp_mean_mag_error',
			'sigg_ps':'g_mean_psf_mag_error',
			'sigr_ps':'r_mean_psf_mag_error',
			'sigi_ps':'i_mean_psf_mag_error',
			'sigz_ps':'z_mean_psf_mag_error',
			'sigy_ps':'y_mean_psf_mag_error',
			'J_2M':'j_m',
			'H_2M':'h_m',
			'Ks_2M':'ks_m',
			'sigJ_2M':'j_msigcom ',
			'sigH_2M':'h_msigcom ',
			'sigKs_2M':'ks_msigcom ',
		}

		# Redenning coefficients
		# from BASE-9 Filters.cpp
		absCoeffs0 = {
         "G":   0.83627 ,
         "G_BP": 1.08337 ,
         "G_RP": 0.63439 ,
         "g_ps": 1.17994 ,
         "r_ps": 0.86190 ,
         "i_ps": 0.67648 ,
         "z_ps": 0.51296 ,
         "y_ps": 0.42905 ,
         "J_2M":  0.28665 ,
         "H_2M":  0.18082 ,
         "Ks_2M": 0.11675 ,
		}
		self.absCoeffs = {}
		for key, value in absCoeffs0.items():
			self.absCoeffs[self.magRenamer[key]] = value



	def getData(self, clusterName, filename):
		columns = ', '.join(self.columns)

		if (self.verbose > 0):
			print("Retrieving Gaia data ... ")

		# for testing 
		# self.SQLcmd = f"SELECT TOP 5 {columns} " + \
		self.ADQLcmd = f"SELECT {columns} " + \
		f"FROM {self.GaiaCatalog} AS gaia " + \
		f"LEFT OUTER JOIN {self.PanSTARRSMatchCatalog} AS best ON gaia.source_id = best.source_id " +  \
		f"LEFT OUTER JOIN {self.PanSTARRSCatalog} AS ps ON best.original_ext_source_id = ps.obj_id " +  \
		f"LEFT OUTER JOIN {self.TMASSMatchCatalog} AS xmatch ON gaia.source_id = xmatch.source_id " + \
		f"LEFT OUTER JOIN {self.TMASSJoinCatalog} AS xjoin ON xmatch.clean_tmass_psc_xsc_oid = xjoin.clean_tmass_psc_xsc_oid " + \
		f"LEFT OUTER JOIN {self.TMASSCatalog} AS tmass ON xjoin.original_psc_source_id = tmass.designation " + \
		f"WHERE CONTAINS(POINT('ICRS',gaia.ra, gaia.dec)," + \
		f"CIRCLE('ICRS', {self.RA}, {self.Dec}, {self.radius}))=1 " + \
		f"AND abs(gaia.pmra_error)<{self.maxPMerror} " + \
		f"AND abs(gaia.pmdec_error)<{self.maxPMerror} " + \
		f"AND gaia.pmra IS NOT NULL AND abs(gaia.pmra)>0 " + \
		f"AND gaia.pmdec IS NOT NULL AND abs(gaia.pmdec)>0 " + \
		f"AND gaia.parallax > (1000 /{self.max_distance});"

		print (self.ADQLcmd)

		if (self.verbose > 1):
			print(self.ADQLcmd)
		if self.radius < 3:
			job = Gaia.launch_job_async(self.ADQLcmd, dump_to_file=False) #could save this to a file
			self.data = job.get_results()

			# calculate the photometric errors 
			# from here: https://cdsarc.unistra.fr/viz-bin/ReadMe/I/350?format=html&tex=true#sRM3.63
			# found via here: https://astronomy.stackexchange.com/questions/38371/how-can-i-calculate-the-uncertainties-in-magnitude-like-the-cds-does
			sigmaG_0 = 0.0027553202
			sigmaGBP_0 = 0.0027901700
			sigmaGRP_0 = 0.0037793818
			self.data['phot_g_mean_mag_error'] = ((-2.5/np.log(10)/self.data['phot_g_mean_flux_over_error'])**2 + sigmaG_0**2)**0.5
			self.data['phot_bp_mean_mag_error'] = ((-2.5/np.log(10)/self.data['phot_bp_mean_flux_over_error'])**2 + sigmaGBP_0**2)**0.5
			self.data['phot_rp_mean_mag_error'] = ((-2.5/np.log(10)/self.data['phot_rp_mean_flux_over_error'])**2 + sigmaGRP_0**2)**0.5

			self.data['coord'] = SkyCoord(self.data['ra'], self.data['dec'], frame='icrs') 
			self.data['rCenter'] = self.center.separation(self.data['coord'])
			self.data['id'] = self.data['SOURCE_ID']
			self.data['distance'] = (self.data['parallax']).to(units.parsec, equivalencies=units.parallax()).to(units.parsec).value
			self.data['distance_error'] = (self.data['parallax_error']).to(units.parsec, equivalencies=units.parallax()).to(units.parsec).value
			self.data = self.data.to_pandas()
			self.saveDataToFile(filename)
			print ('Data queried and saved.')
		else:
			print (clusterName, 'too large.  Query from website.')
		# self.red = True
		# self.getParallaxMembers(clusterName)
		# print ('Parallax mean =', self.pa_fit.parameters[1])
		# print ('Getting differential reddening E(B-V) values...')
		# bayestar = BayestarWebQuery(version='bayestar2019')
		# reddening = []
		# samples = []
		# for i in range(1,5):
		# 	size = int(len(self.data)/4)
		# 	start = (i-1)*size
		# 	stop = i*size
		# 	if i == 4:
		# 		ra = self.data['ra'][start:]
		# 		dec = self.data['dec'][start:]
		# 	else:
		# 		ra = self.data['ra'][start:stop]
		# 		dec = self.data['dec'][start:stop]
		# 	d = [self.pa_fit.parameters[1]]*len(ra)
		# 	coords = SkyCoord(ra=ra*units.deg, dec=dec*units.deg,
		# 	                   distance=d*units.pc, frame='icrs')
		# 	reddening.append(bayestar(coords, mode='median'))
		# 	percentiles = bayestar(coords, mode='percentile', pct=[16,84])
		# 	samples.append([x[-1]-x[0] for x in percentiles])
		# self.data['sig_E(B-V)'] = np.concatenate(samples)
		# self.data['E(B-V)'] = np.concatenate(reddening)
		# self.saveDataToFile(clusterName+'_dir/'+clusterName+'_GaiaData.ecsv')
		# if (self.verbose > 2):
		# 	print(self.data)




	def saveDataToFile(self, filename,mems_only=False):
		if (self.verbose > 0):
			print(f"Saving data to file {filename} ... ")
		if mems_only == True:
			self.data[self.data['membership']>self.mem_min].to_csv(filename, sep=' ',index=False)
		else:
			self.data.to_csv(filename, sep=' ',index=False)  
	

	def readDataFromFile(self, clusterName, filename=None):
		# read and save the data from an ecsv file
		if (filename is None):
			filename = 'GaiaData.ecsv'

		if (self.verbose > 0):
			print(f"Reading data from file {filename} ... ")
		self.data = pd.read_csv(filename,sep=' ')
		#self.data = ascii.read(filename)  
		# if self.deredden == 1:
		# 	absCoeffs0 = {
		# 		"G":   0.83627 ,
		# 		"G_BP": 1.08337 ,
		# 		"G_RP": 0.63439 ,
		# 		"g_ps": 1.17994 ,
		# 		"r_ps": 0.86190 ,
		# 		"i_ps": 0.67648 ,
		# 		"z_ps": 0.51296 ,
		# 		"y_ps": 0.42905 ,
		# 		"J_2M":  0.28665 ,
		# 		"H_2M":  0.18082 ,
		# 		"Ks_2M": 0.11675 ,
		# 		}
		# 	med_red = np.mean(self.data['E(B-V)'])
		# 	self.data['g_mean_psf_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['g_ps'])
		# 	self.data['r_mean_psf_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['r_ps'])
		# 	self.data['i_mean_psf_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['i_ps'])
		# 	self.data['z_mean_psf_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['z_ps'])
		# 	self.data['y_mean_psf_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['y_ps'])
		# 	self.data['phot_g_mean_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['G'])
		# 	self.data['phot_bp_mean_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['G_BP'])
		# 	self.data['phot_rp_mean_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['G_RP'])
		# 	self.data['j_m']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['J_2M'])
		# 	self.data['h_m']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['H_2M'])
		# 	self.data['ks_m']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['Ks_2M'])

		# 	#add error from reddening in quadrature
		# 	var = np.std(self.data['E(B-V)'])**2
		# 	self.data['g_mean_psf_mag_error']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['g_ps'])**2+self.data['g_mean_psf_mag_error']**2)
		# 	self.data['r_mean_psf_mag_error']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['r_ps'])**2+self.data['r_mean_psf_mag_error']**2)
		# 	self.data['i_mean_psf_mag_error']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['i_ps'])**2+self.data['i_mean_psf_mag_error']**2)
		# 	self.data['z_mean_psf_mag_error']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['z_ps'])**2+self.data['z_mean_psf_mag_error']**2)
		# 	self.data['y_mean_psf_mag_error']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['y_ps'])**2+self.data['y_mean_psf_mag_error']**2)
		# 	self.data['phot_g_mean_mag_error']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['G'])**2+self.data['phot_g_mean_mag_error']**2)
		# 	self.data['phot_bp_mean_mag_error']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['G_BP'])**2+self.data['phot_bp_mean_mag_error']**2)
		# 	self.data['phot_rp_mean_mag_eror']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['G_RP'])**2+self.data['phot_rp_mean_mag_error']**2)
		# 	self.data['j_msigcom']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['J_2M'])**2+self.data['j_msigcom']**2)
		# 	self.data['h_msigcom']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['H_2M'])**2+self.data['h_msigcom']**2)
		# 	self.data['ks_msigcom']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['Ks_2M'])**2+self.data['ks_msigcom']**2)
		# else:
		# 	print ('No reddening  corrections applied')
		# self.data  = Table(self.data, masked=True, copy=False)
		# self.data['coord'] = SkyCoord(self.data['ra'], self.data['dec'], frame='icrs') 
		# self.data['rCenter'] = self.center.separation(self.data['coord'])
		# #self.data['id'] = [int(str(x)[-15:]) for x in self.data['source_id']]
		# self.data['id'] = self.data['source_id']
		# self.data['parallax'] = (self.data['parallax']).to(units.parsec, equivalencies=units.parallax()).to(units.parsec).value
		#if self.pass_no == 1:
		#	self.data = self.data[self.data['rCenter'] <= self.radius]
		#self.data = self.data.drop_duplicates(keep='first')

	def get_small_data(self, clusterName):
		if self.group_no == None:
			check = 'n'
			while check == 'n':
				lim_radius = float(input('Radius for HDBSCAN? (in degrees)'))
				small_data = self.data[self.data['rCenter']<lim_radius]
				blob = small_data[['ra','dec','pmra','pmdec','parallax']]
				min_cluster_size = int(input('Min cluster size?'))
				clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
				clusterer.fit(blob)
				small_data['label'] = clusterer.labels_
				small_data['group_probabilities'] = clusterer.probabilities_
				no_groups = np.max(clusterer.labels_)
				for i in range(-1,no_groups+1):
					group_no = i
					print (len(small_data[small_data['label']==group_no]), ' members in group ',i)
				group_no = int(input('Which group?'))
				group = small_data[small_data['label']==group_no]
				print ("Cluster distance = ", np.median(group['distance'].values)," pc")
				cs=plt.scatter(group['phot_bp_mean_mag']-group['phot_rp_mean_mag'],group['phot_g_mean_mag'],c=group['group_probabilities'])
				low_ylim = min(group['phot_g_mean_mag'])-1
				high_ylim = max(group['phot_g_mean_mag'])+1
				plt.ylim(high_ylim, low_ylim)
				plt.colorbar(cs,pad=0,label='Probability')
				plt.show()
				check = input('Use this group? (y/n)?')
			self.group_no = group_no
			self.lim_radius = lim_radius
			self.min_cluster_size = min_cluster_size
		else:
			small_data = self.data[self.data['rCenter']<self.lim_radius]
			blob = small_data[['ra','dec','pmra','pmdec','parallax']]
			clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
			clusterer.fit(blob)
			small_data['label'] = clusterer.labels_
			small_data['group_probabilities'] = clusterer.probabilities_
			group = small_data[small_data['label']==self.group_no]
			print ("Cluster distance = ", np.median(group['distance'].values)," pc")
			cs=plt.scatter(group['phot_bp_mean_mag']-group['phot_rp_mean_mag'],group['phot_g_mean_mag'],c=group['group_probabilities'])
			low_ylim = min(group['phot_g_mean_mag'])-1
			high_ylim = max(group['phot_g_mean_mag'])+1
			plt.ylim(high_ylim, low_ylim)
			plt.colorbar(cs,pad=0,label='Probability')
			plt.title(clusterName)
			plt.savefig(self.plotNameRoot + clusterName+'_HDBSCAN_cmd.png', bbox_inches='tight')
			plt.show()
		self.small_data = small_data[small_data['label']==self.group_no]

	def get_p_value(self, param):
		self.small_data[param].fill_value = np.nan
		x = self.small_data[param][np.isnan(self.small_data[param])==False]
		if param == 'radial_velocity':
			units = 'km/s'
			df_name = 'PRV'
			bins=int(len(x)/3)
		if param == 'distance':
			units = 'pc'
			df_name = 'PPa'
			bins=int(len(x)/10)
		if param == 'pmra':
			units = 'mas/yr'
			df_name = 'PPMra'
			bins=int(len(x)/10)
		if param == 'pmdec':
			units = 'mas/yr'
			df_name = 'PPMdec'
			bins=int(len(x)/10)

		low_lim = np.percentile(x,5)
		high_lim = np.percentile(x,95)
		hrv, brv = np.histogram(x,bins=bins,range=(low_lim,high_lim))
		#fit
		#guess = brv[np.argmax(hrv)]
		guess = np.mean(x)
		std = np.std(x)
		p_init = models.Gaussian1D(np.max(hrv), guess, std)


		fit_p = self.fitter
		fit = fit_p(p_init, brv[:-1], hrv)
		print (fit.stddev, std)
		if fit.stddev > std*5:
			fit = p_init
		mean = fit.parameters[1]
		sig = fit.parameters[2]
		xf = np.linspace(np.min(x), np.max(x), len(x))

		fig, ax = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
		ax[0].step(brv[:-1],hrv, color='black')
		ax[0].plot(xf,gauss(xf,max(hrv),mean,sig), color='deeppink', lw=3, label=param+ f' = {mean:.1f} '+units)
		ax[0].set_xlim(low_lim, high_lim)
		ax[0].set_title(param+', HDBSCAN members')
		ax[0].legend()

		x = self.data[param][np.isnan(self.data[param])==False]
		hrv, brv = np.histogram(x,bins=int(len(x)/3))
		ax[1].step(brv[:-1],hrv, color='black')
		ax[1].plot(xf,gauss(xf,max(hrv),mean,sig), color='deeppink', lw=3)
		ax[1].set_xlim(low_lim, high_lim)
		ax[1].set_title(param+', all data')

		self.data[df_name] = (self.data[param] - mean)**2/(sig**2)
		fig.savefig(self.plotNameRoot + param +'_hist.png', bbox_inches='tight')
		plt.show()

	def combineMemberships(self):
		if (self.verbose > 0):
			print("combining memberships ...")
		self.data['PPM'] = self.data['PPMra'].fillna(0)+self.data['PPMdec'].fillna(0)
		df = self.data
		chi2 = df['PRV'].fillna(0)+df['PPa'].fillna(0)+self.data['PPMra'].fillna(0)+self.data['PPMdec'].fillna(0)
		#p=1-stats.chi2.cdf(chi2, 3)
		p = stats.chi2.sf(chi2,4)
		self.data['membership'] = p

	def get_coreRadius(self):
		mask = (self.data['membership']>self.mem_min)
		members = self.data[mask]
		bins = np.linspace(0, np.max(members['rCenter'])*60, 50)

		# calculate the bin centers
		bin_centers = bins[:-1] + np.diff(bins)[0]

		# get the radial histogram (number in each bin)
		nr,br = np.histogram(members['rCenter']*60,bins = bins)

		# calculate the surface areas
		sarea = np.array([np.pi*bins[i+1]**2. - np.pi*bins[i]**2. for i in range(len(bins) - 1)])

		# calculate the surface density (N/area)
		sdensity = nr/sarea

		# calculate the uncertainty on the surface density (using propagation of errors and assuming zero uncertainty on the surface area)
		err_sdensity = np.sqrt(nr)/sarea

		# make sure that the errors are always non-zero (not sure the best approach here)
		xx = np.where(err_sdensity == 0)
		err_sdensity[xx] = 1./sarea[xx]

		# fit a King model
		p_init = models.KingProjectedAnalytic1D(max(sdensity), np.median(bin_centers), float(self.rt)*60)
		fit_p = self.fitter
		king = fit_p(p_init, bin_centers, sdensity, weights = 1.0/err_sdensity)

		# get the parameters and the uncertainties
		# https://github.com/astropy/astropy/issues/7202
		param_names = king.param_names
		params = king.parameters
		try:
			err_params = np.sqrt(np.diag(fit_p.fit_info['param_cov']))
		except:
			err_params = params
			print ('Problem with err_params.')

		print('fit parameters :')
		for (x1,x2,x3) in zip(param_names, params, err_params):
		    print(f'{x1} = {x2:.3} +/- {x3:.3}')


		# plot with the king model fit
		f,ax = plt.subplots()
		_ = ax.errorbar(bin_centers, sdensity, yerr = err_sdensity, fmt = '.')
		foo = models.KingProjectedAnalytic1D(*king.parameters)
		ax.plot(bin_centers, foo(bin_centers), color='black',ls='--')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.set_xlabel('r [arcmin]')
		ax.set_ylabel(r'N / arcmin$^2$')
		plt.show()



	# 	if (self.createPlots):
	# 		f = plt.figure(figsize=(8, 8)) 
	# 		gs = gridspec.GridSpec(2, 2, height_ratios = [1, 3], width_ratios = [3, 1]) 
	# 		ax1 = plt.subplot(gs[0])
	# 		ax2 = plt.subplot(gs[2])
	# 		ax3 = plt.subplot(gs[3])

	# 		#histograms
	# 		hx1D, x1D = np.histogram(x,bins=self.PMxbins,range=(self.PMxmin,self.PMxmax))
	# 		xf = np.linspace(self.PMxmin,self.PMxmax,self.PMxbins)
	# 		ax1.step(x1D[:-1], hx1D, color='black')
	# 		#ax1.plot(xf, tri_norm(xf, *self.x_params[0]), color='deeppink', lw=3)
	# 		ax1.plot(xf, gauss(xf, *self.x_params[0][:3]), color='deeppink')
	# 		ax1.plot([xmean, xmean],[0,max(hx1D)],lw=2,color='gray',ls='dotted')
	# 		ax1.plot([xmean-self.sig_fac*xsig, xmean-self.sig_fac*xsig],[0,max(hx1D)],lw=3,color='gray')
	# 		ax1.plot([xmean+self.sig_fac*xsig, xmean+self.sig_fac*xsig],[0,max(hx1D)],lw=3,color='gray')
	# 		#ax1.plot(xf, bi_norm(xf, *x_params[0][:6]), color='darkslateblue', ls='dashed')
	# 		#ax1.axvline(self.x_params[0][0],color='tab:purple', ls='dotted')
	# 		ax1.annotate(r'$\mu_\alpha$ =' + str(round(xmean,2)) + r'mas yr$^{-1}$', (xmean+1, .9*max(hx1D)),fontsize=15) 
	# 		ax1.set_ylim(-.05*max(hx1D),max(hx1D)+.1*max(hx1D))

	# 		hy1D, y1D = np.histogram(y,bins=self.PMybins,range=(self.PMymin,self.PMymax))
	# 		yf = np.linspace(self.PMymin,self.PMymax,200)
	# 		ax3.step(hy1D, y1D[:-1], color='black')
	# 		#ax3.plot(tri_norm(yf, *self.y_params[0]),yf, color='deeppink', lw=3)
	# 		ax3.plot(gauss(yf, *self.y_params[0][0:3]),yf, color='deeppink')

	# 		ax3.plot([0,max(hy1D)],[ymean-self.sig_fac*ysig, ymean-self.sig_fac*ysig],lw=3,color='gray')
	# 		ax3.plot([0,max(hy1D)],[ymean+self.sig_fac*ysig, ymean+self.sig_fac*ysig],lw=3,color='gray')

	# 		#ax3.plot(bi_norm(yf, *y_params[0][:6]),yf, color='darkslateblue', ls='dashed')
	# 		ax3.axhline(self.y_params[0][0],color='tab:purple', ls='dotted')
	# 		ax3.plot([0,max(hy1D)], [ymean, ymean],lw=2,color='gray',ls='dotted')
	# 		ax3.annotate(r'$\mu_\alpha$ =' + str(round(ymean,2)) + r'mas yr$^{-1}$',(0.85*max(hy1D), ymean+1), rotation=270,fontsize=15)
	# 		ax3.set_xlim(-.05*max(hy1D),max(hy1D)+.1*max(hy1D))

	# 		ax2.scatter(x,y,color='darkgray',marker='.')
	# 		ax2.scatter(x[self.data['PPM'] > 0], y[self.data['PPM'] > 0], color='cyan', marker='.')

	# 		ax1.set_xlim(self.PMxmin, self.PMxmax)
	# 		ax2.set_xlim(self.PMxmin, self.PMxmax)
	# 		ax2.set_ylim(self.PMymin, self.PMymax)
	# 		ax3.set_ylim(self.PMymin, self.PMymax)
	# 		ax2.set_xlabel(r'$\mu_\alpha$ (mas yr$^{-1}$)', fontsize=16)
	# 		ax2.set_ylabel(r'$\mu_\delta$ (mas yr$^{-1}$)', fontsize=16)
	# 		plt.setp(ax1.get_yticklabels()[0], visible=False)
	# 		plt.setp(ax1.get_xticklabels(), visible=False)
	# 		plt.setp(ax3.get_yticklabels(), visible=False)
	# 		plt.setp(ax3.get_xticklabels()[0], visible=False)
	# 		f.subplots_adjust(hspace=0., wspace=0.)
	# 		ax1.set_title(clusterName)
	# 		if (savefig):
	# 			f.savefig(self.plotNameRoot + 'PMHist'+str(self.pass_no)+'.pdf', format='PDF', bbox_inches='tight')






	def plotCMD(self, data=None, x1='g_mean_psf_mag', x2='i_mean_psf_mag', y='g_mean_psf_mag', m='membership', savefig=True):
		if (self.verbose > 0):
			print("plotting CMD ...")

		if (data is None):
			data = self.data

		# I could specify the columns to use
		f, ax = plt.subplots(figsize=(5,8))
		ax.plot(data[x1] - data[x2], data[y],'.', color='lightgray')

		#members
		mask = self.data[self.data['membership'] > self.mem_min]
		#self.get_minMembership(data[mask][y])
		ax.plot(mask[x1] - mask[x2], mask[y],'.', color='deeppink')
		ax.set_ylim(max(mask[y]), min(mask[y])-.5)
		#ax.set_xlim(-1, 5)
		ax.set_xlabel('G_BP'+'-'+'G_RP', fontsize=16)
		ax.set_ylabel('G', fontsize=16)
		if (savefig):
			f.savefig(self.plotNameRoot + 'CMD.pdf', format='PDF', bbox_inches='tight')
		plt.show()

	def generatePhotFile(self):
		if (self.verbose > 0):
			print("generating phot file ...")
		# create a *.phot file for input to BASE-9
		# would be nice if this was more general and could handle any set of photometry

		# take only those that pass the membership threshold
		members = self.data.loc[self.data['membership'] > self.mem_min].copy(deep=True)
		#members = self.data.copy(deep=True)
		print ('Length of members = ', len(members))
		#members['membership'] = 0.01
		# members['membership'] = ((members['membership']-min(members['membership']))/(max(members['membership'])-min(members['membership']))) \
		# * 0.8 + 0.1

		# include only the columns we need in the output table.
		# Currently I am not including Gaia photometry
		# If we want to include Gaia photometry, we need to include errors.  
		# Maybe we can use "typical errors" from here: https://gea.esac.esa.int/archive/documentation/GEDR3/index.html
		# add the extra columns for BASE-9
		members['mass1'] = 1.1 #if we know masses, these could be added
		members['massRatio'] = 0.0 #if we know mass ratios, these could be added
		members['stage1'] = 1 #set to 1 for MS and giant stars (use 2(?) for WDs)
		members['useDBI'] = 1 #set to 1 to use during burn-in.  May want to improve to remove anomalous stars
		out = members[['id', 
					   'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
					   'g_mean_psf_mag', 'r_mean_psf_mag', 'i_mean_psf_mag', 'z_mean_psf_mag', 'y_mean_psf_mag',
					   'j_m', 'h_m', 'ks_m',
					   'phot_g_mean_mag_error', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error',
					   'g_mean_psf_mag_error', 'r_mean_psf_mag_error', 'i_mean_psf_mag_error', 'z_mean_psf_mag_error', 'y_mean_psf_mag_error',
					   'j_msigcom', 'h_msigcom', 'ks_msigcom',
					   'mass1', 'massRatio', 'stage1','membership','useDBI'
					   ]].copy(deep=True)
		# rename columns
		out = out.rename(columns={'phot_g_mean_mag':'G'})
		out = out.rename(columns={'phot_bp_mean_mag':'G_BP'}) 
		out = out.rename(columns={'phot_rp_mean_mag':'G_RP'})
		out = out.rename(columns={'g_mean_psf_mag':'g_ps'})
		out = out.rename(columns={'r_mean_psf_mag':'r_ps'})
		out = out.rename(columns={'i_mean_psf_mag':'i_ps'})
		out = out.rename(columns={'z_mean_psf_mag':'z_ps'})
		out = out.rename(columns={'y_mean_psf_mag':'y_ps'})
		out = out.rename(columns={'phot_g_mean_mag_error':'sigG'})
		out = out.rename(columns={'phot_bp_mean_mag_error':'sigG_BP'}) 
		out = out.rename(columns={'phot_rp_mean_mag_error':'sigG_RP'})
		out = out.rename(columns={'g_mean_psf_mag_error':'sigg_ps'})
		out = out.rename(columns={'r_mean_psf_mag_error':'sigr_ps'})
		out = out.rename(columns={'i_mean_psf_mag_error':'sigi_ps'})
		out = out.rename(columns={'z_mean_psf_mag_error':'sigz_ps'})
		out = out.rename(columns={'y_mean_psf_mag_error':'sigy_ps'})
		out = out.rename(columns={'j_m':'J_2M'})
		out = out.rename(columns={'h_m':'H_2M'})
		out = out.rename(columns={'ks_m':'Ks_2M'})
		out = out.rename(columns={'j_msigcom':'sigJ_2M'})
		out = out.rename(columns={'h_msigcom':'sigH_2M'})
		out = out.rename(columns={'ks_msigcom':'sigKs_2M'})
		out = out.rename(columns={'membership':'CMprior'})

		out['CMprior'] = np.where(out['CMprior'] < self.photSigFloor, self.photSigFloor, out['CMprior']) #set a phot CMprior floor of 0.01, per previous experiments
		out['CMprior'] = np.where(out['CMprior'] > 0.9, 0.9, out['CMprior']) #set a phot CMprior ceiling of 0.9 to allow wiggle room for error in the models

		# impose a floor to phot error to be safe
		for c in ['sigG', 'sigG_BP', 'sigG_RP', 'sigg_ps', 'sigr_ps', 'sigi_ps', 'sigz_ps', 'sigy_ps', 'sigJ_2M', 'sigH_2M', 'sigKs_2M']:
			out[c] = np.where((out[c] < 0.02), 0.02, out[c]) #phot error floor of 0.02 is just what works from previous trys

		# replace any nan or mask values with -9.9 for sig, which BASE9 will ignore
		for c in ['G', 'G_BP', 'G_RP', 'g_ps', 'r_ps', 'i_ps', 'z_ps', 'y_ps', 'J_2M', 'H_2M', 'Ks_2M']:
			out[c] = out[c].fillna(99.9)			
		for c in ['sigG', 'sigG_BP', 'sigG_RP', 'sigg_ps', 'sigr_ps', 'sigi_ps', 'sigz_ps', 'sigy_ps', 'sigJ_2M', 'sigH_2M', 'sigKs_2M']:
			out[c] = out[c].fillna(-9.9)


		# expose this so it can be used elsewhere
		self.members = members

		# write the phot file
		self.dumpPhotFile(out)

	


	def dumpPhotFile(self, out, filename=None):
		if (filename is None):
			filename = self.photOutputFileName

		idint = list(map(int, out['id']))
		zfillN = int(12) 

		# write to file with proper formatting
		# fdec = np.abs(np.log10(self.photSigFloor)).astype(int)
		# ffmt = '%-' + str(fdec + 3) + '.' + str(fdec) + 'f'
		ffmt = '%-7.4f'
		with open(filename, 'w', newline='\n') as f:
			ascii.write(Table.from_pandas(out), delimiter=' ', output=f, format = 'basic', overwrite=True,
				formats = {
						'G': ffmt, 'G_BP': ffmt, 'G_RP': ffmt, 
						'g_ps': ffmt, 'r_ps': ffmt, 'i_ps': ffmt, 'z_ps': ffmt, 'y_ps': ffmt, 
						'J_2M': ffmt, 'H_2M': ffmt, 'Ks_2M': ffmt,
						'sigG': ffmt, 'sigG_BP': ffmt, 'sigG_RP': ffmt, 
						'sigg_ps': ffmt, 'sigr_ps': ffmt, 'sigi_ps': ffmt, 'sigz_ps': ffmt, 'sigy_ps': ffmt, 
						'sigJ_2M': ffmt, 'sigH_2M': ffmt, 'sigKs_2M': ffmt,
						'mass1': '%-5.3f', 'massRatio': '%-5.3f', 'stage1': '%1i','CMprior': '%-5.3f','useDBI': '%1d'
						}
						)
			


	def generateYamlFile(self):
		if (self.verbose > 0):
			print("generating yaml file ...")

		# create a base9.yaml file for input to BASE-9

		with open(self.yamlTemplateFileName, 'r') as file:
			yamlOutput = yaml.safe_load(file)

		# to allow for quotes in the output (this may not be necessary)
		# https://stackoverflow.com/questions/38369833/pyyaml-and-using-quotes-for-strings-only
		class quoted(str):
			pass
		def quoted_presenter(dumper, data):
			return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
		yaml.add_representer(quoted, quoted_presenter)
		# remove the "null" output (again, I'm not sure this is necessary)
		# https://stackoverflow.com/questions/37200150/can-i-dump-blank-instead-of-null-in-yaml-pyyaml
		def represent_none(self, _):
			return self.represent_scalar('tag:yaml.org,2002:null', '')
		yaml.add_representer(type(None), represent_none)

		yamlOutput['general']['files']['photFile'] = quoted(self.yamlInputDict['photFile'])
		yamlOutput['general']['files']['outputFileBase'] = quoted(self.yamlInputDict['outputFileBase'])
		yamlOutput['general']['files']['modelDirectory'] = quoted(self.yamlInputDict['modelDirectory'])
		yamlOutput['general']['files']['scatterFile'] = quoted("")
		
		yamlOutput['general']['main_sequence']['msRgbModel'] = int(self.yamlInputDict['msRgbModel'])

		yamlOutput['general']['cluster']['starting']['Fe_H'] = float(self.yamlInputDict['Fe_H'][0])
		yamlOutput['general']['cluster']['starting']['Av'] = float(self.yamlInputDict['Av'][0])
		yamlOutput['general']['cluster']['starting']['Y'] = float(self.yamlInputDict['Y'][0])
		yamlOutput['general']['cluster']['starting']['carbonicity'] = float(self.yamlInputDict['carbonicity'][0])
		yamlOutput['general']['cluster']['starting']['logAge'] = float(self.yamlInputDict['logAge'][0])
		yamlOutput['general']['cluster']['starting']['distMod'] = float(self.yamlInputDict['distMod'][0])

		yamlOutput['general']['cluster']['priors']['means']['Fe_H'] = float(self.yamlInputDict['Fe_H'][1])
		yamlOutput['general']['cluster']['priors']['means']['Av'] = float(self.yamlInputDict['Av'][1])
		yamlOutput['general']['cluster']['priors']['means']['Y'] = float(self.yamlInputDict['Y'][1])
		yamlOutput['general']['cluster']['priors']['means']['carbonicity'] = float(self.yamlInputDict['carbonicity'][1])
		yamlOutput['general']['cluster']['priors']['means']['logAge'] = float(self.yamlInputDict['logAge'][1])
		yamlOutput['general']['cluster']['priors']['means']['distMod'] = float(self.yamlInputDict['distMod'][1])
		#yamlOutput['general']['cluster']['priors']['means']['distMod'] = float(self.dmod)


		yamlOutput['general']['cluster']['priors']['sigmas']['Fe_H'] = float(self.yamlInputDict['Fe_H'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['Av'] = float(self.yamlInputDict['Av'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['Y'] = float(self.yamlInputDict['Y'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['carbonicity'] = float(self.yamlInputDict['carbonicity'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['logAge'] = float(self.yamlInputDict['logAge'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['distMod'] = float(self.yamlInputDict['distMod'][2])
		#yamlOutput['general']['cluster']['priors']['sigmas']['distMod'] = float(self.err_dmod)


		# I hope this doesn't need to be sorted in the same order as the original
		# This outputs in alphabetical order
		with open(self.yamlOutputFileName, 'w') as file:
			yaml.dump(yamlOutput, file, indent = 4)


	def createInteractiveCMD(self, mag = 'G', color1 = 'G_BP', color2 = 'G_RP', xrng = [0.5,2], yrng = [20,10]):
		# NOTE: currently this code requires a column in the data labelled as 'membership'
		# create the initial figure
		TOOLS = "box_zoom, reset, lasso_select, box_select"
		p = figure(title = "",
			tools = TOOLS, width = 500, height = 700,
			x_range = xrng, y_range = yrng)

		
		mask = (self.data['membership'] >= self.mem_min) 
		membershipOrg = self.data['membership'].data.copy() # in case I need to reset
		# add an index column so that I can map back to the original data
		self.data['index'] = np.arange(0,len(self.data))

		sourcePhot = ColumnDataSource(data = dict(x = self.data[mask][color1] - self.data[mask][color2], y = self.data[mask][mag], index = self.data[mask]['index']))
		# empty for now, but will be filled below in updateUseDBI
		sourcePhotSingles = ColumnDataSource(data = dict(x = [] , y = []))

		# add the phot points to the plot
		# Note: I could handle categorical color mapping with factor_cmap, but this does not seem to update in the callback when I change the status in sourcePhot (I removed status since this doesn't work)
		# colorMapper = factor_cmap('status', palette = ['black', 'dodgerblue'], factors = ['unselected', 'selected'])
		photRenderer = p.scatter(source = sourcePhot, x = 'x', y = 'y', alpha = 0.5, size = 3, marker = 'circle', color = 'red')
		p.scatter(source = sourcePhotSingles, x = 'x', y = 'y', alpha = 0.75, size = 8, marker = 'circle', color = 'dodgerblue')

		# add the PointDrawTool to allow users to draw points interactively
		# to hold the user-added points
		newPoints = ColumnDataSource(data = dict(x = [], y = []))   

		renderer = p.circle(source = newPoints, x = 'x', y = 'y', color = 'limegreen', size = 10) 
		drawTool = PointDrawTool(renderers = [renderer])
		p.add_tools(drawTool)
		p.toolbar.active_tap = drawTool

		# add the line connecting the user-added points
		p.line(source = newPoints, x = 'x', y = 'y', color = 'limegreen', width = 4)

		# callback to update the single-star selection when a point is added or when the slider changes
		# https://stackoverflow.com/questions/47177493/python-point-on-a-line-closest-to-third-point

		def updateUseDBI(attr, old, new):
			if (len(newPoints.data['x']) > 1):
				# define the user-drawn line using shapely
				lne = shLs([ (x,y) for x,y in zip(newPoints.data['x'], newPoints.data['y']) ])
				# find the distance for each point to the user-drawn line
				# clear the singles data
				data = dict(x = [] , y = [])
				for i, row in enumerate(self.data):
					if (mask[i]):
						x = row[color1] - row[color2]
						y = row[mag]
						pte = shPt(x, y)
						dst = pte.distance(lne)
						# if the distance is within the user-defined tolerance, set the status to selected
						self.data[i]['useDBI'] = 0
						if (dst < slider.value):
							self.data[i]['useDBI'] = 1
							data['x'].append(x)
							data['y'].append(y)
				sourcePhotSingles.data = data
		newPoints.on_change('data', updateUseDBI)

		##########################
		# widgets

		# add a slider to define the width of the selection, next to the line
		slider = Slider(start = 0, end = 0.1, value = 0.01, step = 0.001, format = '0.000', title = "Selection Region")
		def sliderCallback(attr, old, new):
			updateUseDBI(attr, old, new)
		slider.on_change("value", sliderCallback)
		# add a reset button
		resetButton = Button(label = "Reset",  button_type = "danger", )
		def resetCallback(event):
			newPoints.data = dict(x = [], y = [])
			slider.value = 0.01  
			self.data['useDBI'] = [0]*len(self.data)
			self.data['membership'] = membershipOrg
			mask = (self.data['membership']>=self.mem_min)
			sourcePhot.data = dict(x = self.data[mask][color1] - self.data[mask][color2], y = self.data[mask][mag], index = self.data[mask]['index'])
		resetButton.on_click(resetCallback)

		# add a button to write the files
		writeButton = Button(label = "Write BSS file",  button_type = "success")

		def writeCallback(event):
			# output an updated phot file
			# This will be improved when the code is combined with BASE9_utils
			#self.generatePhotFile(self.sig1photOutputFileName)
			#self.generateYamlFile()
			print('need to write to file ') 

		writeButton.on_click(writeCallback)


		# add a button to delete selected points
		deleteButton = Button(label = "Delete selected points",  button_type = "warning")

		def deleteCallback(event):
			# set the membership to -1, redefine the mask, and remove them from the columnDataSource
			if (len(sourcePhot.selected.indices) > 0):
				indices = sourcePhot.data['index'][sourcePhot.selected.indices]
				self.data['membership'][indices] = -1
				mask = (self.data['membership']>=self.mem_min)
				#mask = (((self.data['PRV'] >= self.RV_memmin) & (self.data['PPa'] >= self.Pa_memmin)) & (self.data['PPM'] >= self.PM_memmin))
				sourcePhot.data = dict(x = self.data[mask][color1] - self.data[mask][color2], y = self.data[mask][mag], index = self.data[mask]['index'])
				# reset
				sourcePhot.selected.indices = []

		deleteButton.on_click(deleteCallback)

		##########################
		# layout
		# plot on the left, buttons on the right
		buttons = column(
			slider, 
			#Div(text='<div style="height: 15px;"></div>'),
			deleteButton,
			#outfile,
			writeButton,
			Div(text='<div style="height: 15px;"></div>'),
			resetButton,
		)
		title = 	Div(text='<div style="font-size:20px; font-weight:bold">Interactive CMD</div>')
		instructions = 	Div(text='<ul style="font-size:14px">\
			<li>Use the lasso or box select tool to select stars that should be removed from the sample completely. Then click the "Delete selected points button"</li>\
			<li>Add points that define a line along the single-star sequence.  To add points, first enable the "Point Draw Tool".  Then click on the plot to create a marker.  Click+drag any marker to move it.  Click+backspace any marker to delete it.</li>\
			<li>Click the "Reset" button to remove all the points and reset the plot. </li>\
			<li>When finished, click the "Apply Selection" button to select the single members.</li>\
		</ul>')

		layout = column(title, instructions, row(p, buttons))
		return(layout)

#####################################
	### For the interactive isochrone ###
	#####################################
	def getModelGrid(self, isochroneFile):
		# get the model grid space [age, FeH] from the model file

		FeH = -99.
		age = -99.
		i = 0
		with open(isochroneFile, 'r') as f:
			for line in f:
				if line.startswith('%s'):
					x = line.replace('=',' ').split()
					FeH = float(x[2])
				if line.startswith('%a'):
					x = line.replace('=',' ').split()
					age = float(x[2])

				if (FeH != -99 and age != -99):
					if (line.startswith('%s') or line.startswith('%a')):
						if (i == 0):
							grid = np.array([age, FeH])
						else:
							grid = np.vstack((grid, [age, FeH]))
						i += 1

		return grid


	def interpolateModel(self, age, FeH, isochroneFile, mag = 'phot_g_mean_mag', color1 = 'phot_bp_mean_mag', color2 = 'phot_rp_mean_mag'):
		# perform a linear interpolation in the model grid between the closest points

		# get the model grid
		grid = self.getModelGrid(isochroneFile)
		# check that the age and Fe/H values are within the grid 
		ageGrid = np.sort(np.unique(grid[:,0]))
		FeHGrid = np.sort(np.unique(grid[:,1]))
		maxAge = np.max(ageGrid)
		minAge = np.min(ageGrid)
		maxFeH = np.max(FeHGrid)
		minFeH = np.min(FeHGrid)
		try:
			# find the 4 nearest age and Fe/H values
			iAge0 = np.where(ageGrid < age)[0][-1]
			age0 = ageGrid[iAge0]
			age1 = age0
			if (iAge0 + 1 < len(ageGrid)):
				age1 = ageGrid[iAge0 + 1]

			iFeH0 = np.where(FeHGrid < FeH)[0][-1]
			FeH0 = FeHGrid[iFeH0]
			FeH1 = FeH0
			if (iFeH0 + 1< len(FeHGrid)):
				FeH1 = FeHGrid[iFeH0 + 1]
			# read in those parts of the isochrone file
			inAge = False
			inFeH = False
			arrS = []
			columns = ''
			testFeH = '-99'
			testAge = '-99'
			with open(isochroneFile, 'r') as f:
				for line in f:
					if (inAge and inFeH and not line.startswith('%') and not line.startswith('#')):
						key = '[' + testAge + ',' + testFeH + ']'
						x = line.strip() + ' ' + testAge + ' ' + testFeH
						arrS.append(x.split())
					if line.startswith('%s'):
						inFeH = False
						x = line.replace('=',' ').split()
						testFeH = x[2]
						if (float(testFeH) == FeH0 or float(testFeH) == FeH1):
							inFeH = True
					if line.startswith('%a'):
						inAge = False
						x = line.replace('=',' ').split()
						testAge = x[2]
						if (float(testAge) == age0 or float(testAge) == age1):
							inAge = True
					if (line.startswith('# EEP')):
						x = line.replace('# ','') + ' logAge' + ' Fe_H'
						columns = x.split()

			# convert this to a pandas dataframe 
			df = pd.DataFrame(arrS, columns = columns, dtype = float)
			df.rename(columns = self.magRenamer, inplace = True)

			# take only the columns that we need
			# We cannot interpolate on mass since that is not unique and not strictly ascedning
			# Ideally we would interpolate on initial mass, but that is not included in the BASE-9 model files
			# I will interpolate on EEP, which I *think* is a number that is unique for each star 
			df = df[np.unique(['EEP', mag, color1, color2, 'logAge','Fe_H'])]
			ages = df['logAge'].unique()
			FeHs = df['Fe_H'].unique()
			EEPs = df['EEP'].unique()

			# initialize the output dataframe
			results = Table()

			# create an array to interpolate on
			# https://stackoverflow.com/questions/30056577/correct-usage-of-scipy-interpolate-regulargridinterpolator
			pts = (ages, FeHs, np.sort(EEPs))
			for arr in np.unique([mag, color1, color2]):
				val_size = list(map(lambda q: q.shape[0], pts))
				vals = np.zeros(val_size)
				for i, a in enumerate(ages):
					for j, f in enumerate(FeHs):
						df0 = df.loc[(df['logAge'] == a) & (df['Fe_H'] == f)]
						interp = interp1d(df0['EEP'], df0[arr], bounds_error = False)
						vals[i,j,:] = interp(EEPs)

				interpolator = RegularGridInterpolator(pts, vals)
				results[arr] = interpolator((age, FeH, EEPs))


			return results

		except:
			print(f"!!! ERROR: could not interpolate isochrone.  Values are likely outside of model grid !!!\nage : {age} [{minAge}, {maxAge}]\nFeH: {FeH} [{minFeH}, {maxFeH}]")
			return None


	def createInteractiveIsochrone(self, isochroneFile, initialGuess = [4, 0, 0, 0], mag = 'phot_g_mean_mag', color1 = 'phot_bp_mean_mag', color2 = 'phot_rp_mean_mag', xrng = [0.5,6], yrng = [20,10], yamlSigmaFactor = 2.0):
		'''
		To run this in a Jupyter notebook (intended purpose):
		--------------------
		layout = createInteractiveIsochrone('isochrone.model', [log10(age), FeH, mM, Av])
		def bkapp(doc):
			doc.add_root(layout)
		show(bkapp)
		'''

		###########################


		# create the initial figure
		TOOLS = "box_zoom, reset, lasso_select, box_select"
		p = figure(title = "",
			tools = TOOLS, width = 500, height = 700,
			x_range = xrng, y_range = yrng)




		self.df = self.data
		self.table = Table.from_pandas(self.df)
		self.table['index'] = np.arange(0, len(self.data))
		mask = (self.df['membership'] >= self.mem_min)
		membershipOrg = self.df['membership'] # in case I need to reset
		# add an index column so that I can map back to the original data
		self.df['index'] = np.arange(0, len(self.df))
		# get the isochrone at the desired age and metallicity
		iso = self.interpolateModel(initialGuess[0], initialGuess[1], isochroneFile, mag = mag, color1 = color1, color2 = color2)

		# convert to observed mags given distance modulus and Av for these filters
		# taken from BASE-9 Star.cpp
		# abs is input Av
		# distance is input distance modulus
		# combinedMags[f] += distance;
		# combinedMags[f] += (evoModels.absCoeffs[f] - 1.0) * clust.abs;
		def offsetMag(magVal, magCol, mM, Av):
			return magVal + mM + (self.absCoeffs[magCol] - 1.)*Av

		def getObsIsochrone(iso, mM, Av):
			color1Obs = offsetMag(iso[color1], color1, mM, Av)
			color2Obs = offsetMag(iso[color2], color2, mM, Av)
			magObs = offsetMag(iso[mag], mag, mM, Av)
			return {'x': color1Obs - color2Obs, 'y': magObs, color1: iso[color1], color2: iso[color2], mag: iso[mag]}



		# define the input for Bokeh
		sourcePhot = ColumnDataSource(data = dict(x = self.df[mask][color1] - self.df[mask][color2], y = self.df[mask][mag], index = self.df[mask]['index']))
		mask = (self.table['membership'] >= self.mem_min)
		oldsourcePhot = ColumnDataSource(data = dict(x = self.table[mask][color1] - self.table[mask][color2], y = self.table[mask][mag], index = self.table[mask]['index']))
		sourceCluster = ColumnDataSource(data = dict(logAge = [initialGuess[0]], Fe_H = [initialGuess[1]], distMod = [initialGuess[2]], Av = [initialGuess[3]]))
		sourceIso = ColumnDataSource(getObsIsochrone(iso, sourceCluster.data['distMod'][0], sourceCluster.data['Av'][0]))

		# add the photometry and isochrone to the plot
		photRenderer = p.scatter(source = sourcePhot, x = 'x', y = 'y', alpha = 0.5, size = 3, marker = 'circle', color = 'black')
		isoRenderer =  p.scatter(source = sourceIso, x = 'x', y = 'y', color = 'red', size = 5)


		###########################
		# widgets

		# text boxes to define the age and FeH value for the isochrone
		# adding a bit of extra code to make the label and input side-by-side
		ageInput = TextInput(value = str(10.**initialGuess[0]/1.e6), title = '')
		ageInputTitle = Paragraph(text = 'age [Myr]:', align = 'center', width = 60)
		ageInputLayout = row([ageInputTitle, ageInput])
		FeHInput = TextInput(value = str(initialGuess[1]), title = '')
		FeHInputTitle = Paragraph(text = '[Fe/H]:', align = 'center', width = 60)
		FeHInputLayout = row([FeHInputTitle, FeHInput])

		# botton to update the isochrone
		updateIsochroneButton = Button(label = "Update isochrone",  button_type = "success")
		def updateIsochroneCallback(event):
			iso = self.interpolateModel(np.log10(float(ageInput.value)*1e6), float(FeHInput.value), isochroneFile, mag = mag, color1 = color1, color2 = color2)
			if (iso is not None):
				sourceCluster.data['logAge'] = [np.log10(float(ageInput.value)*1e6)]
				sourceCluster.data['Fe_H'] = [float(FeHInput.value)]
				sourceIso.data = getObsIsochrone(iso, sourceCluster.data['distMod'][0], sourceCluster.data['Av'][0])
			else:
				ageInput.value = str(10.**sourceCluster.data['logAge'][0]/1.e6)
				FeHInput.value = str(sourceCluster.data['Fe_H'][0])
		updateIsochroneButton.on_click(updateIsochroneCallback)


		# add sliders to move the isochrone in mM and redenning
		mMSlider = Slider(start = 0, end = 20, value = initialGuess[2], step = 0.01, format = '0.00', title = "Distance Modulus")
		def mMSliderCallback(attr, old, new):
			sourceCluster.data['distMod'] = [mMSlider.value]
			iso = sourceIso.data
			sourceIso.data = getObsIsochrone(iso, sourceCluster.data['distMod'][0], sourceCluster.data['Av'][0])
		mMSlider.on_change("value", mMSliderCallback)

		AvSlider = Slider(start = 0, end = 3, value = initialGuess[3], step = 0.001, format = '0.000', title = "Av")
		def AvSliderCallback(attr, old, new):
			sourceCluster.data['Av'] = [AvSlider.value]
			iso = sourceIso.data
			sourceIso.data = getObsIsochrone(iso, sourceCluster.data['distMod'][0], sourceCluster.data['Av'][0])
		AvSlider.on_change("value", AvSliderCallback)


		# add a button to delete selected points
		deleteButton = Button(label = "Delete selected points",  button_type = "danger")

		def deleteCallback(event):
			# set the membership to -1, redefine the mask, and remove them from the columnDataSource
			indices = oldsourcePhot.data['index'][sourcePhot.selected.indices]
			self.table['membership'][indices] = -1
			del_ids = self.table['id'][self.table['index'][indices]]
			self.df['membership'][self.df['id'].isin(del_ids)] = -1
			mask = (self.df['membership'] >= self.mem_min )
			sourcePhot.data = dict(x = self.df[mask][color1] - self.df[mask][color2], y = self.df[mask][mag], index = self.df[mask]['index'])
			# reset
			mask1 = (self.table['membership'] > self.mem_min)
			oldsourcePhot.data = dict(x = self.table[mask1][color1] - self.table[mask1][color2], y = self.table[mask1][mag], index = self.table[mask1]['index'])
			sourcePhot.selected.indices = []
			indices = sourcePhot.data['index'][sourcePhot.selected.indices]
		

		deleteButton.on_click(deleteCallback)

		# add a reset button
		resetButton = Button(label = "Reset",  button_type = "warning", )

		def resetCallback(event):
			self.df['membership'] = membershipOrg
			mask = (self.df['membership'] >= self.mem_min)
			sourcePhot.data = dict(x = self.df[mask][color1] - self.df[mask][color2], y = self.df[mask][mag], index = self.df[mask]['index'])

		resetButton.on_click(resetCallback)


		# add a button to write the files
		writeButton = Button(label = "Write .phot and .yaml files",  button_type = "success")

		def writeCallback(event):
			# output updated phot files
			self.generatePhotFile()

			# update the yaml starting values and prior variances if necessary
			print('initial and final yaml [starting, mean, sigma] values:')
			keys = ['Fe_H', 'Av', 'distMod', 'logAge']
			for k in keys:
				# initial values
				init = self.yamlInputDict[k].copy()
				self.yamlInputDict[k][0] = sourceCluster.data[k][0]

				#variances
				mean = self.yamlInputDict[k][1]
				sig = self.yamlInputDict[k][2]
				val = sourceCluster.data[k][0]
				minVal = mean - sig
				maxVal = mean + sig
				if (val <= minVal or val >= maxVal):
					diff = abs(val - mean)
					self.yamlInputDict[k][2] = yamlSigmaFactor*diff

				print(f'{k}: initial = {init}, final = {self.yamlInputDict[k]}')
				if np.abs(val-mean) > sig:
					print ('WARNING:',k,' sig value too small.  Increasing by 0.25')
					self.yamlInputDict[k][2] += 0.25
			self.generateYamlFile()
			print('Files saved : ', self.photOutputFileName, self.yamlOutputFileName) 


		writeButton.on_click(writeCallback)

		###########################
		# layout
		# plot on the left, buttons on the right


		buttons = column(
			Div(text='<div style="height: 15px;"></div>'),
			ageInputLayout,
			FeHInputLayout,
			updateIsochroneButton,
			mMSlider,
			AvSlider,
			deleteButton,
			resetButton,
			Div(text='<div style="height: 50px;"></div>'),
			writeButton,
		)
		title = 	Div(text='<div style="font-size:20px; font-weight:bold">Interactive CMD</div>')
		instructions = 	Div(text='<ul style="font-size:14px">\
			<li>To delete points: select points with lasso or box select tool, and click the "Delete" button.</li>\
			<li>Click the "Reset" button undo all delete actions. </li>\
			<li>To change the isochrone, enter the age and FeH in the appropriate boxes, and click the "Update Isochrone" button.</li>\
			<li>Move the isochrone to change the distance and redenning to better match the data.</li>\
			<li>When finished, click the "Write files" button output the results.</li>\
		</ul>')


		layout = column(title, instructions, row(p, buttons))
		return(layout)


	def query_data(self, clusterName, filename):
		def distance_modulus_to_pc(distance_modulus):
			return 10 ** ((distance_modulus + 5) / 5)
		try:
			OCdf = pd.read_csv('OCcompiled_clean_v2.csv')
			row = OCdf.loc[OCdf['ID'] == clusterName].iloc[0]
			self.row = row
			# get the cluster center
			self.center = SkyCoord(row['RA[hr]'], row['Dec[deg]'], unit=(units.hourangle, units.degree))
			self.RA = self.center.ra.to(units.degree).value
			self.Dec = self.center.dec.to(units.degree).value
			#self.center = center.ra.to(units.degree).value
			#calculate the distance modulus and error
			dmod = 5.*np.log10(row['dist[pc]']) - 5.
			err_dmod = (5.*1./np.log(10.)*(1./row['dist[pc]']))*row['err_dist[pc]'] # error propagation
			# estimate the cluster tidal radius 
			# equation from Binney and Tremaine for Jacoby Radius (8.91)
			# want this in degrees
			Mg = 1.5*10**12 #*units.solMass #this probably could use verification and a reference
			self.rt = row['rgc[pc]']*(row['mass[Msun]']/(3.*Mg))**(1./3.)
			pc = distance_modulus_to_pc(dmod)
			self.max_distance = pc + 1000
		except:
			Hunt = pd.read_csv('Hunt2023.tsv',sep='\s+', header=94)
			Hunt_data = Hunt[Hunt['Name']==clusterName]
			dmod = Hunt_data['MOD50'].values[0]
			ra = Hunt_data['RA_ICRS'].values[0]
			dec = Hunt_data['DE_ICRS'].values[0]
			self.center = SkyCoord(ra, dec, unit=(units.degree, units.degree))
			self.RA = self.center.ra.to(units.degree).value
			self.Dec = self.center.dec.to(units.degree).value
			self.rt = float(Hunt_data['rt'].values[0])
			pc = distance_modulus_to_pc(dmod)
			self.max_distance = pc + 1000
		rtfac = 2
		self.radius = min(rtfac*self.rt, 3)
		print ('Getting data from a radius of',self.radius,' deg')
		self.getData(clusterName, filename)
		return



	def runAll(self, clusterName, filename=None):
		self.readDataFromFile(clusterName, filename)
		self.get_small_data(clusterName)
		params = ["radial_velocity", "distance", "pmra", "pmdec"]
		[self.get_p_value(param) for param in params]
		self.combineMemberships()
		self.plotCMD(y = 'phot_g_mean_mag', x1 = 'phot_bp_mean_mag', x2 = 'phot_rp_mean_mag')
		self.generatePhotFile()
		self.generateYamlFile()
		self.get_coreRadius()
		self.saveDataToFile('OC_data/'+clusterName+'_dir/'+clusterName+'.csv',mems_only=True)
		plt.close('all')
		print("done.")
