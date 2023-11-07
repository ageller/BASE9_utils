# Python code to download Gaia data at a given location

from astroquery.gaia import Gaia
from astropy.modeling.models import KingProjectedAnalytic1D
from astropy.modeling import models, fitting
from astropy.table import Table, Column, vstack, MaskedColumn
import astropy.units as units
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Table, join, hstack


import scipy.stats
import scipy.optimize
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import quad
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import yaml
import pandas as pd
import os

from bokeh import *
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Button, PointDrawTool, Slider, TextInput, Div, Paragraph, PanTool, BoxSelectTool
from bokeh.layouts import column, row
from shapely.geometry import LineString as shLs
from shapely.geometry import Point as shPt

from dustmaps.bayestar import BayestarWebQuery



def gauss(x, A, mu, sigma): #single Gaussian function
	return A*np.exp(-(x-mu)**2/(2*sigma**2))

def bi_norm(x, *args):  #double Gaussian function
	A1, m1, s1, A2, m2,s2 = args
	ret = gauss(x,A1,m1,s1)+gauss(x,A2,m2,s2)
	return ret
def tri_norm(x, *args):  #triple Gaussian function
	A1, m1, s1, A2, m2,s2, A3, m3,s3 = args
	ret = A1*scipy.stats.norm.pdf(x, loc=m1 ,scale=s1)
	ret += A2*scipy.stats.norm.pdf(x, loc=m2 ,scale=s2)
	ret += A3*scipy.stats.norm.pdf(x, loc=m3 ,scale=s3)
	return ret


class GaiaClusterMembers(object):
	'''
	This Class will grab data from the Gaia archive.  The user must provide the RA and Dec values,
	and the Class will return the full catalog and 

	RA, Dec and radius should be provided in decimal degrees.
	
	'''
	
	def __init__(self, RA = None, Dec = None, radius = 1, **kwargs): #Query Gaia catalog
		
		# required inputs
		self.RA = RA # degrees
		self.Dec = Dec # degrees

		# radius of the area to return
		self.radius = radius # degrees

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


		self.deredden = True #To apply reddening corrections from the Bayestar dustmap, set self.deredden =True.  Otherwise, set self.deredden=False.

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


		self.photSigFloor = 0.02 # floor to the photometry errors for the .phot file
		# output
		self.SQLcmd = ''
		self.data = None # will be an astropy table
		self.createPlots = True # set to True to generate plots
		self.plotNameRoot = ''
		self.photOutputFileName = 'input.phot'
		self.yamlOutputFileName = 'base9.yaml'
		self.saveDataFile = 'cluster.ecsv'

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

		# multiple passes are typically used to narrow in on the cluster members
		# these are defined in runAll
		self.pass_no = 1

		# default is to include stars within 10 sigma bounds around the cluster fit mean
		self.sig_fac = 10
	
	def getData(self, clusterName):
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
		f"AND gaia.pmdec IS NOT NULL AND abs(gaia.pmdec)>0;"


		print (self.ADQLcmd)

		if (self.verbose > 1):
			print(self.ADQLcmd)
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
		self.data['id'] = self.data['source_id']
		self.data['distance'] = (self.data['parallax']).to(units.parsec, equivalencies=units.parallax()).to(units.parsec).value
		self.data['distance_error'] = (self.data['parallax_error']).to(units.parsec, equivalencies=units.parallax()).to(units.parsec).value
		self.data = self.data.to_pandas()
		self.red = True
		self.getParallaxMembers(clusterName)
		print ('Parallax mean =', self.pa_fit.parameters[1])
		print ('Getting differential reddening E(B-V) values...')
		bayestar = BayestarWebQuery(version='bayestar2019')
		reddening = []
		samples = []
		for i in range(1,5):
			size = int(len(self.data)/4)
			start = (i-1)*size
			stop = i*size
			if i == 4:
				ra = self.data['ra'][start:]
				dec = self.data['dec'][start:]
			else:
				ra = self.data['ra'][start:stop]
				dec = self.data['dec'][start:stop]
			d = [self.pa_fit.parameters[1]]*len(ra)
			coords = SkyCoord(ra=ra*units.deg, dec=dec*units.deg,
							   distance=d*units.pc, frame='icrs')
			reddening.append(bayestar(coords, mode='median'))
			percentiles = bayestar(coords, mode='percentile', pct=[16,84])
			samples.append([x[-1]-x[0] for x in percentiles])
		self.data['sig_E(B-V)'] = np.concatenate(samples)
		self.data['E(B-V)'] = np.concatenate(reddening)
		self.saveDataToFile(self.saveDataFile)
		if (self.verbose > 2):
			print(self.data)




	def saveDataToFile(self, filename=None):
		# save the data to an ecsv file
		if (filename is None):
			filename = 'GaiaData.ecsv'

		if (self.verbose > 0):
			print(f"Saving data to file {filename} ... ")
		self.data.to_csv(filename, sep=' ',index=False)  


	def readDataFromFile(self, filename=None):
		# read and save the data from an ecsv file
		if (filename is None):
			filename = 'GaiaData.ecsv'

		if (self.verbose > 0):
			print(f"Reading data from file {filename} ... ")
		self.data = pd.read_csv(filename,sep=' ')
		if self.deredden:
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
			med_red = np.mean(self.data['E(B-V)'])
			self.data['g_mean_psf_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['g_ps'])
			self.data['r_mean_psf_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['r_ps'])
			self.data['i_mean_psf_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['i_ps'])
			self.data['z_mean_psf_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['z_ps'])
			self.data['y_mean_psf_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['y_ps'])
			self.data['phot_g_mean_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['G'])
			self.data['phot_bp_mean_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['G_BP'])
			self.data['phot_rp_mean_mag']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['G_RP'])
			self.data['j_m']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['J_2M'])
			self.data['h_m']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['H_2M'])
			self.data['ks_m']-=(med_red-self.data['E(B-V)'])*3.1*(absCoeffs0['Ks_2M'])

			#add error from reddening in quadrature
			var = np.std(self.data['E(B-V)'])**2
			self.data['g_mean_psf_mag_error']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['g_ps'])**2+self.data['g_mean_psf_mag_error']**2)
			self.data['r_mean_psf_mag_error']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['r_ps'])**2+self.data['r_mean_psf_mag_error']**2)
			self.data['i_mean_psf_mag_error']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['i_ps'])**2+self.data['i_mean_psf_mag_error']**2)
			self.data['z_mean_psf_mag_error']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['z_ps'])**2+self.data['z_mean_psf_mag_error']**2)
			self.data['y_mean_psf_mag_error']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['y_ps'])**2+self.data['y_mean_psf_mag_error']**2)
			self.data['phot_g_mean_mag_error']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['G'])**2+self.data['phot_g_mean_mag_error']**2)
			self.data['phot_bp_mean_mag_error']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['G_BP'])**2+self.data['phot_bp_mean_mag_error']**2)
			self.data['phot_rp_mean_mag_eror']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['G_RP'])**2+self.data['phot_rp_mean_mag_error']**2)
			self.data['j_msigcom']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['J_2M'])**2+self.data['j_msigcom']**2)
			self.data['h_msigcom']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['H_2M'])**2+self.data['h_msigcom']**2)
			self.data['ks_msigcom']=np.sqrt((self.data['sig_E(B-V)']**2+var/len(self.data))*(3.1*absCoeffs0['Ks_2M'])**2+self.data['ks_msigcom']**2)
		else:
			print ('No reddening  corrections applied')
		if self.pass_no == 1:
			self.data = self.data[self.data['rCenter'] <= self.radius]

		self.data = self.data.drop_duplicates(keep='first')


	def get_coreRadius(self):
		mask = (self.data['membership']>0)
		members = self.data[mask]

		# get the histogram of radial values
		# set the radial bins (in arcmin)
		bins = np.linspace(0, int(self.radius*60), 60)

		# calculate the bin centers
		bin_centers = bins[:-1] + np.diff(bins)[0]

		# get the radial histogram (number in each bin)
		nr,br = np.histogram(members['rCenter'],bins = bins)
		# calculate the surface areas
		sarea = np.array([bins[i+1]**2. - bins[i]**2. for i in range(len(bins) - 1)])

		# calculate the surface density (N/area)
		sdensity = nr/sarea

		# calculate the uncertainty on the surface density (using propagation of errors and assuming zero uncertainty on the surface area)
		err_sdensity = np.sqrt(nr)/sarea

		# make sure that the errors are always non-zero (not sure the best approach here)
		xx = np.where(err_sdensity == 0)
		err_sdensity[xx] = 1./sarea[xx]
		
		# fit a King model
		p_init = models.KingProjectedAnalytic1D(max(sdensity), np.median(bin_centers), self.rt)
		fit_p = self.fitter
		king = fit_p(p_init, bin_centers, sdensity, weights = 1.0/err_sdensity)
		param_names = king.param_names
		params = king.parameters
		self.r_core = float(params[1])/60.
		self.r_tide = float(params[2])/60.


	def getRVMembers(self, clusterName, savefig=True):
		# calculate radial-velocity memberships
		if (self.verbose > 0):
			print("Finding radial-velocity members ... ")
		
		self.data['radial_velocity'].fill_value = np.nan
		x = self.data['radial_velocity']
		#1D histogram
		hrv, brv = np.histogram(x, bins = self.RVbins, range=(self.RVmin, self.RVmax))

		#fit
		RVguess = brv[np.argmax(hrv)]
		if (self.RVmean is not None):
			RVguess = self.RVmean
		if self.pass_no ==1:
			p_init = models.Gaussian1D(np.max(hrv), RVguess, 1) \
					+ models.Gaussian1D(5, brv[np.argmax(hrv)], 50)
			if clusterName == 'M_35':
				p_init = models.Gaussian1D(np.max(hrv), -10, 20) +\
				models.Gaussian1D(np.max(hrv), RVguess, 10)


			fit_p = self.fitter
			rvG1D = fit_p(p_init, brv[:-1], hrv)
			self.rv_fit = rvG1D
		if self.pass_no ==2:
			rvG1D = self.rv_fit

		if (self.verbose > 1):
			print(rvG1D)

		mean = self.rv_fit.parameters[1]
		sig = self.rv_fit.parameters[2]
		RV_sigfac = self.sig_fac

		self.data['PRV'] = (self.data['radial_velocity'] - mean)**2/(sig**2)
		self.data['PRV'] = np.where(((self.data['radial_velocity'] > mean+RV_sigfac*sig) & (np.isnan(self.data['radial_velocity'])==False)) | ((self.data['radial_velocity'] < mean-RV_sigfac*sig) & (np.isnan(self.data['radial_velocity'])==False)), 0, self.data['PRV'])
		if clusterName == 'M_21' or clusterName == 'NGC_2264':
			self.data['PRV'] = 1
		if (self.createPlots):

			font = {'family' : 'Times New Roman',
				'weight' : 'normal',
				'size'   : 15}
			plt.rc('font', **font)
			plt.rc('text', usetex=True)
			fig = plt.figure(figsize=(10,10))
			fig.patch.set_color('white')
			hrv, brv = np.histogram(x, bins = self.RVbins, range=(self.RVmin, self.RVmax))
			xf = np.linspace(self.RVmin, self.RVmax, self.RVbins*10)
			fig, ax = plt.subplots()
			ax.step(brv[:-1],hrv, color='black')
			ax.plot(xf,bi_norm(xf,*rvG1D.parameters), color='gray',lw=3)
			plt.plot([mean-RV_sigfac*sig,mean-RV_sigfac*sig],[0,max(hrv)],color='gray',lw=3)
			plt.plot([mean+RV_sigfac*sig,mean+RV_sigfac*sig],[0,max(hrv)],color='gray',lw=3)
			plt.plot()
			foo = models.Gaussian1D(*rvG1D.parameters[0:3])
			ax.plot(xf, foo(xf), color='deeppink', lw=3)
			foo = models.Gaussian1D(*rvG1D.parameters[3:])
			ax.plot(xf, foo(xf), color='darkslateblue', ls='dashed')
			ax.set_xlabel(r'RV (km/s', fontsize = 30)
			ax.set_ylabel('N', fontsize = 30)
			ax.axvline(rvG1D.parameters[1], color='tab:purple', ls='dotted')
			ax.annotate(f'RV = {rvG1D.parameters[1]:.1f} km/s', (rvG1D.parameters[1] + 0.05*(self.RVmax - self.RVmin), 0.95*max(hrv)) )

			if (savefig):
 				fig.savefig(os.path.join(self.plotNameRoot, 'RVHist'+str(self.pass_no)+'.pdf'), format='PDF', bbox_inches='tight')
			


	def getParallaxMembers(self, clusterName, savefig=True):
		# estimate memberships based on distance (could use Bailer Jones, but this simply uses inverted parallax)
		if (self.verbose > 0):
			print("Finding parallax members ... ")
		x = self.data['distance']
		hpa, bpa = np.histogram(x, bins = self.dbins, range=(self.dmin, self.dmax))

		#fit
		dguess = bpa[np.argmax(hpa)]
		if (self.distance is not None):
			dguess = self.distance
		p_init = models.Gaussian1D(np.max(hpa), dguess, 10)\
				+ models.Polynomial1D(degree=self.dPolyD)
		fit_p = self.fitter
		pa1D = fit_p(p_init, bpa[:-1], hpa)
		if self.pass_no == 1:
		 	self.pa_fit = pa1D
		if (self.verbose > 1):
			print(pa1D)
		if clusterName == 'M_35' and self.pass_no==1:
			#AMP,MEAN,SIG
			p_init = models.Gaussian1D(4.26940535e+02,  8.49163992e+02,  4.03301980e+01) + models.Polynomial1D(degree=6)
			pa1D = fit_p(p_init, bpa[:-1], hpa)
			pa1D.parameters[2]= 8.49163992e+02
		if self.pass_no == 2:
			pa1D = self.pa_fit
		mean = pa1D.parameters[1]
		sig = pa1D.parameters[2]
		self.data['PPa'] = (self.data['distance'] - mean)**2/(sig**2)
		if clusterName != 'NGC_6791':
			self.data['PPa'] = np.where(((self.data['distance'] > mean+self.sig_fac*sig)  & (np.isnan(self.data['distance'])==False)) \
			 | ((self.data['distance'] < mean-self.sig_fac*sig) & (np.isnan(self.data['distance'])==False)), 0, self.data['PPa'])
		else:
			self.data['PPa'] = 1
		if (self.createPlots):
			hpa, bpa = np.histogram(x, bins = self.dbins, range=(self.dmin, self.dmax))
			xf = np.linspace(self.dmin, self.dmax, self.dbins*10)
			f,ax = plt.subplots()
			ax.step(bpa[:-1],hpa, color='black')
			ax.plot(xf,pa1D(xf), color='gray', lw=3)
			if clusterName == 'NGC_6791':
				ax.plot(xf,pa1D(xf), color='gray')
			else:
				foo = models.Gaussian1D(*pa1D.parameters[0:3])
				ax.plot(xf, foo(xf), color='deeppink', lw=3)
			ax.plot([mean-self.sig_fac*sig, mean-self.sig_fac*sig],[0,max(hpa)],color='gray',lw=3)
			ax.plot([mean+self.sig_fac*sig, mean+self.sig_fac*sig],[0,max(hpa)],color='gray',lw=3)
			foo = models.Polynomial1D(degree=self.dPolyD)
			ax.plot(xf, foo.evaluate(xf,*pa1D.parameters[3:]), color='darkslateblue', ls='dashed')
			ax.set_xlabel('Distance (pc)', fontsize = 30)
			ax.set_ylabel('N', fontsize = 30)
			ax.set_title('NGC 2682', fontsize=30)
			ax.axvline(pa1D.parameters[1], color='tab:purple', ls='dotted')
			ax.annotate(f'd = {pa1D.parameters[1]:.1f} pc', (pa1D.parameters[1] + 0.1*(self.dmax - self.dmin), 0.95*max(hpa)), fontsize=20)
			if (savefig):
				f.savefig(os.path.join(self.plotNameRoot, 'dHist'+str(self.pass_no)+'.pdf'), format='PDF', bbox_inches='tight')




	def getPMMembers(self, clusterName, savefig=True):
		if (self.verbose > 0):
			print("finding proper-motion members ...")

		def getPMparams(x, type):
		#1D histogram
			if clusterName == 'M_35':
				mask1 = (self.data['phot_g_mean_mag'] < 15)
				x = x[mask1]
			if clusterName == 'NGC_7789':
				mask1 = (self.data['phot_g_mean_mag'] < 18)
				x = x[mask1]
			if type == 'x':
				hrv1, brv1 = np.histogram(x, bins = self.PMxbins, range=(self.PMxmin, self.PMxmax))
			if type == 'y':
				hrv1, brv1 = np.histogram(x, bins = self.PMxbins, range=(self.PMymin, self.PMymax))
			max_peak = brv1[np.argmax(hrv1)]
			if self.pass_no == 1:
				left = max([brv1[i] for i in range(len(hrv1)) if brv1[i] < max_peak and hrv1[i] <= len(x)/1000])
				right = min([brv1[i] for i in range(len(hrv1)) if brv1[i] > max_peak and hrv1[i] <= len(x)/1000]) #find range for fit
			if self.pass_no == 2:
				if type == 'x':
					left = self.PMxmin
					right = self.PMxmax
				if type == 'y':
					left = self.PMymin
					right = self.PMymax
			hrv, brv = np.histogram(x, bins = self.PMxbins, range=(left, right))
			params = [np.max(hrv),brv[np.argmax(hrv)], .1,np.max(hrv),brv[np.argmax(hrv)], 1]
			fitted_params,_ = scipy.optimize.curve_fit(bi_norm,brv[:-1], hrv, p0=params, method='lm',sigma=np.sqrt(hrv))
			xf = np.linspace(left, right, len(hrv))
			synth_dist = bi_norm(xf, *fitted_params)
			residuals =[hrv[i]-synth_dist[i] for i in range(len(hrv))]
			res_max_peak = brv[np.argmax(residuals)]
			res_params=[hrv[np.argmax(residuals)],res_max_peak,1]
			if self.pass_no == 1:
				tri_params = np.append(res_params, fitted_params)
				if type == 'x':
					self.pm_xparams = tri_params
				if type == 'y':
					self.pm_yparams = tri_params
			if self.pass_no == 2:
				if type == 'x':
					tri_params = self.pm_xparams
				if type == 'y':
					tri_params = self.pm_yparams
					print (tri_params)

			tri_fitted_params,_ = scipy.optimize.curve_fit(tri_norm,brv[:-1], hrv, p0=tri_params, method='lm',sigma=np.sqrt(hrv))
			sigmas = [tri_fitted_params[i] for i in [2,5,8]]
			G1_sig = [i for i in [2,5,8] if tri_fitted_params[i] == min(sigmas)][0]
			G1_params =[tri_fitted_params[G1_sig-2],tri_fitted_params[G1_sig-1],tri_fitted_params[G1_sig]]
			bi_norm_params = [tri_fitted_params[i] for i in range(len(tri_fitted_params)) if i not in [G1_sig-2,G1_sig-1,G1_sig]]
			tri_fitted_params = np.append(G1_params,bi_norm_params)
			if self.pass_no == 1 and type == 'x':
				self.PMxmin = left
				self.PMxmax = right
			if self.pass_no == 2 and type == 'x':
				left = self.PMxmin
				right = self.PMxmax
			if self.pass_no == 1 and type == 'y':
				self.PMymin = left
				self.PMymax = right
			if self.pass_no == 2 and type == 'y':
				left = self.PMymin
				right = self.PMymax
			return tri_fitted_params,left,right

		x = self.data['pmra']
		y = self.data['pmdec']
		if self.pass_no == 1:
			self.x_params = getPMparams(x,'x')
			self.y_params = getPMparams(y,'y')
			self.PMxmin=self.x_params[1]
			self.PMxmax=self.x_params[2]
			self.PMymin=self.y_params[1]
			self.PMymax=self.y_params[2]		
			self.PM_ra_fit = self.x_params[0][:3]
			self.PM_dec_fit = self.y_params[0][:3]

		xmean = self.PM_ra_fit[1]
		xsig = self.PM_ra_fit[2]
		ymean = self.PM_dec_fit[1]
		ysig = self.PM_dec_fit[2]

		self.data['PM_ra']=(self.data['pmra']- xmean)**2/(xsig**2) 
		self.data['PM_dec']=(self.data['pmdec'] - ymean)**2/(ysig**2)
		self.data['PM_ra'] = np.where((self.data['pmra'] > xmean+self.sig_fac*xsig) | (self.data['pmra'] < xmean-self.sig_fac*xsig), 0, self.data['PM_ra'])
		self.data['PM_dec']=np.where((self.data['pmdec'] > ymean+self.sig_fac*ysig) | (self.data['pmdec'] < ymean-self.sig_fac*ysig), 0, self.data['PM_dec'])
		PPM1 = self.data['PM_ra']*self.data['PM_dec']
		PPM2 =self.data['PM_ra']+self.data['PM_dec']
		self.data['PPM'] = PPM1

		self.combineMemberships()
		if (self.verbose > 1):
			print(x_params,y_params)
		if (self.createPlots):
			fig = plt.figure(figsize=(8, 8)) 
			gs = gridspec.GridSpec(2, 2, height_ratios = [1, 3], width_ratios = [3, 1]) 
			ax1 = plt.subplot(gs[0])
			ax2 = plt.subplot(gs[2])
			ax3 = plt.subplot(gs[3])

			#histograms
			hx1D, x1D = np.histogram(x,bins=self.PMxbins,range=(self.PMxmin,self.PMxmax))
			xf = np.linspace(self.PMxmin,self.PMxmax,self.PMxbins)
			ax1.step(x1D[:-1], hx1D, color='black')
			#ax1.plot(xf, tri_norm(xf, *self.x_params[0]), color='deeppink', lw=3)
			ax1.plot(xf, gauss(xf, *self.x_params[0][:3]), color='deeppink')
			ax1.plot([xmean, xmean],[0,max(hx1D)],lw=2,color='gray',ls='dotted')
			ax1.plot([xmean-self.sig_fac*xsig, xmean-self.sig_fac*xsig],[0,max(hx1D)],lw=3,color='gray')
			ax1.plot([xmean+self.sig_fac*xsig, xmean+self.sig_fac*xsig],[0,max(hx1D)],lw=3,color='gray')
			ax1.annotate(r'$\mu_\alpha$ =' + str(round(xmean,2)) + r' (mas yr$^{-1}$)', (xmean+2, .9*max(hx1D)),fontsize=20) 
			ax1.set_ylim(-.05*max(hx1D),max(hx1D)+.1*max(hx1D))

			hy1D, y1D = np.histogram(y,bins=self.PMybins,range=(self.PMymin,self.PMymax))
			yf = np.linspace(self.PMymin,self.PMymax,200)
			ax3.step(hy1D, y1D[:-1], color='black')
			ax3.plot(gauss(yf, *self.y_params[0][0:3]),yf, color='deeppink')

			ax3.plot([0,max(hy1D)],[ymean-self.sig_fac*ysig, ymean-self.sig_fac*ysig],lw=3,color='gray')
			ax3.plot([0,max(hy1D)],[ymean+self.sig_fac*ysig, ymean+self.sig_fac*ysig],lw=3,color='gray')

			ax3.axhline(self.y_params[0][0],color='tab:purple', ls='dotted')
			ax3.plot([0,max(hy1D)], [ymean, ymean],lw=2,color='gray',ls='dotted')
			ax3.annotate(r'$\mu_\delta$ =' + str(round(ymean,2)) + r' (mas yr$^{-1}$)',(0.85*max(hy1D), -23), rotation=270,fontsize=20)
			ax3.set_xlim(-.05*max(hy1D),max(hy1D)+.1*max(hy1D))


			# cs = ax2.hist2d(x,y,bins=(500,500),norm=mpl.colors.LogNorm(),cmap=plt.cm.Blues,label='Count')#, range=[[-1,1],[-1,1]]) #These two lines will produce a 2D-hist rather than the scatter plot below
			# fig.colorbar(cs[3],ax=ax3,label='N')

			ax2.scatter(x,y,color='darkgray',marker='.')
			ax2.scatter(x[self.data['PPM'] > 0], y[self.data['PPM'] > 0], color='cyan', marker='.')


			ax1.set_xlim(self.PMxmin, self.PMxmax)
			ax2.set_xlim(self.PMxmin, self.PMxmax)
			ax2.set_ylim(self.PMymin, self.PMymax)
			ax3.set_ylim(self.PMymin, self.PMymax)
			ax2.set_xlabel(r'$\mu_\alpha$ (mas yr$^{-1}$)', fontsize=30)
			ax2.set_ylabel(r'$\mu_\delta$ (mas yr$^{-1}$)', fontsize=30)
			plt.setp(ax1.get_yticklabels()[0], visible=False)
			plt.setp(ax1.get_xticklabels(), visible=False)
			plt.setp(ax3.get_yticklabels(), visible=False)
			plt.setp(ax3.get_xticklabels()[0], visible=False)
			fig.subplots_adjust(hspace=0., wspace=0.)

			if (savefig):
				fig.savefig(os.path.join(self.plotNameRoot, 'PMHist'+str(self.pass_no)+'.pdf'), format='PDF', bbox_inches='tight')


	def combineMemberships(self):
		if (self.verbose > 0):
			print("combining memberships ...")

		df = self.data
		chi2 = df['PRV'].fillna(0)+df['PPa'].fillna(0)+df['PPM'].fillna(0)
		p=1-stats.chi2.cdf(chi2, 3)
		self.data['membership'] = self.data['PRV'].fillna(1)*self.data['PPa'].fillna(1)*self.data['PPM'].fillna(1)
		self.data['membership'] = np.where((self.data['membership']==0),0, p)
		self.data['membership'] = np.where((self.data['membership']>0) & (self.data['membership']< 0.01) ,0.01, self.data['membership'])
		self.data['membership'] = np.where((self.data['membership']>0.99) ,0.99, self.data['membership'])



	def plotCMD(self, data=None, x1='g_mean_psf_mag', x2='i_mean_psf_mag', y='g_mean_psf_mag', m='membership', savefig=True):
		if (self.verbose > 0):
			print("plotting CMD ...")

		if (data is None):
			data = self.data

		# I could specify the columns to use
		f, ax = plt.subplots(figsize=(5,8))
		ax.plot(data[x1] - data[x2], data[y],'.', color='lightgray')

		#members
		mask = self.data[self.data['membership'] > 0]
		ax.plot(mask[x1] - mask[x2], mask[y],'.', color='deeppink')
		ax.set_ylim(max(mask[y]), min(mask[y])-.5)
		ax.set_xlabel('G_BP'+'-'+'G_RP', fontsize=16)
		ax.set_ylabel('G', fontsize=16)
		if (savefig):
			f.savefig(os.path.join(self.plotNameRoot, 'CMD.pdf'), format='PDF', bbox_inches='tight')


	def generatePhotFile(self):
		if (self.verbose > 0):
			print("generating phot file ...")

		# create a *.phot file for input to BASE-9
		# would be nice if this was more general and could handle any set of photometry

		# take only those that pass the membership threshold
		members = self.data.loc[self.data['membership'] > 0].copy(deep=True)
		print ('Length of members = ', len(members))

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

		# impose a floor to phot error to be safe
		for c in ['sigG', 'sigG_BP', 'sigG_RP', 'sigg_ps', 'sigr_ps', 'sigi_ps', 'sigz_ps', 'sigy_ps', 'sigJ_2M', 'sigH_2M', 'sigKs_2M']:
			out[c] = np.where((out[c] < self.photSigFloor), self.photSigFloor, out[c])

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
		#yamlOutput['general']['cluster']['priors']['means']['distMod'] = float(self.yamlInputDict['distMod'][1])
		yamlOutput['general']['cluster']['priors']['means']['distMod'] = float(self.dmod)


		yamlOutput['general']['cluster']['priors']['sigmas']['Fe_H'] = float(self.yamlInputDict['Fe_H'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['Av'] = float(self.yamlInputDict['Av'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['Y'] = float(self.yamlInputDict['Y'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['carbonicity'] = float(self.yamlInputDict['carbonicity'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['logAge'] = float(self.yamlInputDict['logAge'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['distMod'] = float(self.err_dmod)


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

		
		self.data['useDBI'] = [1]*len(self.data)
		mask = (self.data['membership'] > 0) 
		membershipOrg = self.data['membership'].data.copy() # in case I need to reset
		# add an index column so that I can map back to the original data
		self.data['index'] = np.arange(0,len(self.data))

		sourcePhot = ColumnDataSource(data = dict(x = self.data[mask][color1] - self.data[mask][color2], y = self.data[mask][mag], index = self.data[mask]['index']))
		# empty for now, but will be filled below in updateUseDBI
		sourcePhotSingles = ColumnDataSource(data = dict(x = [] , y = []))

		# add the phot points to the plot
		# Note: I could handle categorical color mapping with factor_cmap, but this does not seem to update in the callback when I change the status in sourcePhot (I removed status since this doesn't work)
		# colorMapper = factor_cmap('status', palette = ['black', 'dodgerblue'], factors = ['unselected', 'selected'])
		photRenderer = p.scatter(source = sourcePhot, x = 'x', y = 'y', alpha = 0.5, size = 3, marker = 'circle', color = 'black')
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
			mask = (((self.data['PRV'] >= self.RV_memmin) & (self.data['PPa'] >= self.Pa_memmin)) & (self.data['PPM'] >= self.PM_memmin))
			sourcePhot.data = dict(x = self.data[mask][color1] - self.data[mask][color2], y = self.data[mask][mag], index = self.data[mask]['index'])
		resetButton.on_click(resetCallback)

		# add a button to write the files
		writeButton = Button(label = "Write .phot and .yaml files",  button_type = "success")

		def writeCallback(event):
			# output an updated phot file
			# This will be improved when the code is combined with BASE9_utils
			self.generatePhotFile()
			self.generateYamlFile()
			print('Files saved : ', self.photOutputFileName, self.yamlOutputFileName) 

		writeButton.on_click(writeCallback)


		# add a button to delete selected points
		deleteButton = Button(label = "Delete selected points",  button_type = "warning")

		def deleteCallback(event):
			# set the membership to -1, redefine the mask, and remove them from the columnDataSource
			if (len(sourcePhot.selected.indices) > 0):
				indices = sourcePhot.data['index'][sourcePhot.selected.indices]
				self.data['membership'][indices] = -1
				mask = (((self.data['PRV'] >= self.RV_memmin) & (self.data['PPa'] >= self.Pa_memmin)) & (self.data['PPM'] >= self.PM_memmin))
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
			pts = (ages, FeHs, EEPs)
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


	def createInteractiveIsochrone(self, isochroneFile, initialGuess = [4, 0, 0, 0], mag = 'phot_g_mean_mag', color1 = 'phot_bp_mean_mag', color2 = 'phot_rp_mean_mag', xrng = [0.5,2], yrng = [20,10], yamlSigmaFactor = 2.0):
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
		mask = (self.df['membership'] > 0)
		membershipOrg = self.df['membership'] # in case I need to reset
		# add an index column so that I can map back to the original data
		self.table['index'] = np.arange(0, len(self.data))
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
		mask = (self.table['membership'] > 0)
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
			if (len(sourcePhot.selected.indices) > 0):
				indices = oldsourcePhot.data['index'][sourcePhot.selected.indices]
				self.table['membership'][indices] = -1
				del_ids = self.table['id'][self.table['index'][indices]]
				self.df['membership'][self.df['id'].isin(del_ids)] = -1
				mask = (self.df['membership'] > 0 )
				sourcePhot.data = dict(x = self.df[mask][color1] - self.df[mask][color2], y = self.df[mask][mag], index = self.df[mask]['index'])
				# reset
				mask = (self.table['membership'] > 0 )
				oldsourcePhot.data = dict(x = self.table[mask][color1] - self.table[mask][color2], y = self.table[mask][mag], index = self.table[mask]['index'])
				sourcePhot.selected.indices = []

		deleteButton.on_click(deleteCallback)

		# add a reset button
		resetButton = Button(label = "Reset",  button_type = "warning", )

		def resetCallback(event):
			self.df['membership'] = membershipOrg
			mask = (self.df['membership'] > 0)
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
					print ('WARNING:',k,' sig value too small.  Increasing by 1.')
					self.yamlInputDict[k][2] += 1
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

	def createInteractiveSelector(self, 
		mag = 'phot_g_mean_mag',  
		color1 = 'phot_bp_mean_mag', 
		color2 = 'phot_rp_mean_mag', 
		rvbins = np.linspace(-100,100,200), 
		pmrabins = np.linspace(-10,10,200), 
		pmdecbins = np.linspace(-10,10,200), 
		distbins = np.linspace(0, 4000, 200), 
		cmdxrng = [0.5,2], 
		cmdyrng = [20,10]
	):
		
		"""
		Create an interactive plot with Bokeh that allows users to select regions in RV, PM, distance and see the selected 
		stars on the CMD
		"""
		# calcuate the distance and convert to pandas
		data = self.data.copy()
		data['dist'] = (data['parallax'].values*units.mas).to(units.parsec, equivalencies = units.parallax()).value
		
		limits = ColumnDataSource(
			data = dict(
				rv = [min(rvbins), max(rvbins)],
				pmra = [min(pmrabins), max(pmrabins)],
				pmdec = [min(pmdecbins), max(pmdecbins)],
				dist = [min(distbins), max(distbins)],  
			)
		)
		
		source = ColumnDataSource(
			data = dict(
				rv = data['radial_velocity'],
				pmra = data['pmra'],
				pmdec = data['pmdec'],
				dist = data['dist'],
				mag = data[mag],
				color = data[color1] - data[color2],
			)
		)

		def define_selected():
			usedata = data.loc[
							(((data['radial_velocity'] >= limits.data['rv'][0]) &
								(data['radial_velocity'] <= limits.data['rv'][1])) |
								(pd.isna(data['radial_velocity']))
							) &
							((data['pmra'] >= limits.data['pmra'][0]) & 
								(data['pmra'] <= limits.data['pmra'][1])
							) &
							((data['pmdec'] >= limits.data['pmdec'][0]) & 
								(data['pmdec'] <= limits.data['pmdec'][1])
							) &
							((data['dist'] >= limits.data['dist'][0]) & 
								(data['dist'] <= limits.data['dist'][1])
							)
							]
			s = ColumnDataSource(
				data = dict(
					rv = usedata['radial_velocity'],
					pmra = usedata['pmra'],
					pmdec = usedata['pmdec'],
					dist = usedata['dist'],
					mag = usedata[mag],
					color = usedata[color1] - usedata[color2],
				)
			)
			return(s)
		
		# cmd
		pcmd = figure(title = 'CMD', tools = 'box_zoom,reset,pan', width = 500, height = 840, x_range = cmdxrng, y_range = cmdyrng)
		pcmd.scatter(source = source, x = 'color', y = 'mag', alpha = 0.2, size = 3, marker = 'circle', color = 'gray')
		selected_cmd = pcmd.scatter(source = define_selected(), x = 'color', y = 'mag', alpha = 1, size = 4, marker = 'circle', color = 'red')
		pcmd.xaxis.axis_label = color1 + ' - ' + color2
		pcmd.yaxis.axis_label = mag
		
		
		def make_hist(skey, bins, xlabel, width = 400, height = 210, fill_color = 'gray', line_color = 'black'):
			hist, edges = np.histogram(source.data[skey], bins = bins)
			s = ColumnDataSource(data = dict(hist = hist, left = edges[:-1], right = edges[1:]))
			def on_boxselect(attr, old, new):
				if (len(new) > 0):
					limits.data[skey] = [s.data['left'][new[0]], s.data['right'][new[-1]]]
				else:
					limits.data[skey] = [bins[0], bins[-1]]
				ss = define_selected()
				selected_cmd.data_source.data = dict(ss.data)
				print(limits.data)
			s.selected.on_change('indices', on_boxselect)

			p = figure(width = width, height = height, tools = 'xwheel_zoom,reset', active_scroll = 'xwheel_zoom')
			p.add_tools(PanTool(dimensions = "width"))
			p.add_tools(BoxSelectTool(dimensions = 'width'))
			p.quad(source = s, top = 'hist', bottom = 0, left = 'left', right = 'right', fill_color = fill_color, line_color = line_color)
			p.xaxis.axis_label = xlabel
			p.yaxis.axis_label = 'N'
			
			return(p)
		


		# histograms
		prv = make_hist('rv', rvbins, 'Radial Velocity (km/s)', )
		ppmra = make_hist('pmra', pmrabins, 'Proper Motion RA (mas/yr)', )
		ppmdec = make_hist('pmdec', pmdecbins, 'Proper Motion Dec (mas/yr)')
		pdist = make_hist('dist', distbins,  'Distance (pc)')

		title = Div(text='<div style="font-size:20px; font-weight:bold">Interactively Select Cluster Members</div>')
		instructions = Div(text='<div style="font-size:14px">Use the "Box Select" tool in any/all of the histograms to locate the likely cluster members.  After a selection is made in any histogram, the CMD updates to show the slected stars in red, and the current limits are printed below the plots. </div>')


		# construct the interactive
		layout = column(title, instructions, row(column(prv, ppmra, ppmdec, pdist), pcmd))

		return (layout)


	def runAll(self, clusterName, filename=None):
		OCdf = pd.read_csv('OCcompiled_clean_v2.csv')
		row = OCdf.loc[OCdf['ID'] == clusterName].iloc[0]
		self.row = row
		
		# get the cluster center
		self.center = SkyCoord(row['RA[hr]'], row['Dec[deg]'], unit=(units.hourangle, units.degree))
		self.RA = self.center.ra.to(units.degree).value
		self.Dec = self.center.dec.to(units.degree).value
		
		# estimate the cluster tidal radius 
		# equation from Binney and Tremaine for Jacoby Radius (8.91)
		# want this in degrees
		Mg = 1.5*10**12 #*units.solMass #this probably could use verification and a reference
		self.rt = row['rgc[pc]']*(row['mass[Msun]']/(3.*Mg))**(1./3.)
		self.pass_no = 1
		if filename is not None: #read in exisiting data file
			rtfac = 1
			self.radius = rtfac*np.arctan2(self.rt, row['dist[pc]'])*180./np.pi
			print ('Radius=', self.radius)
			self.readDataFromFile(filename = filename)
		else: #if no exisiting data file, query data from a radius of self.radius
			rtfac = 6
			self.radius = rtfac*np.arctan2(self.rt, row['dist[pc]'])*180./np.pi
			print ('Getting data from a radius of',self.radius,' deg')
			self.getData(clusterName)
			print ('Data queried and saved.')
		self.getRVMembers(clusterName)
		self.getParallaxMembers(clusterName)
		self.dmod = 5.*np.log10(self.pa_fit[0].mean[0]) - 5.
		self.err_dmod = (5.*1./np.log(10.)*(1./self.pa_fit[0].mean[0]))*self.pa_fit[0].stddev[0]
		self.getPMMembers(clusterName) #Memberships combined in PM functions
		self.get_coreRadius()
		self.pass_no = 2
		self.readDataFromFile(filename = filename)
		print ('Radius=',max(self.data['rCenter']))
		self.getRVMembers(clusterName)
		self.getParallaxMembers(clusterName)
		self.getPMMembers(clusterName)
		self.get_coreRadius()
		self.plotCMD(y = 'phot_g_mean_mag', x1 = 'phot_bp_mean_mag', x2 = 'phot_rp_mean_mag')
		self.generatePhotFile()
		self.generateYamlFile()
		print ('Final r_core=',self.r_core)
		self.saveDataToFile(self.saveDataFile)
		if (self.verbose > 0):
			print("done.")
