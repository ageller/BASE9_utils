# Python code to download Gaia data at a given location

from astroquery.gaia import Gaia
from astropy.modeling import models, fitting
from astropy.table import Table, Column, vstack, MaskedColumn
import astropy.units as units
from astropy.coordinates import SkyCoord
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
from bokeh.models import ColumnDataSource, Button, Div, PointDrawTool, Slider, TextInput
from bokeh.layouts import column, row
from bokeh.io import curdoc
from bokeh.transform import factor_cmap

from shapely.geometry import LineString as shLs
from shapely.geometry import Point as shPt

def gauss(x, mu, sigma, A):
	return A*np.exp(-(x-mu)**2/2/sigma**2)
def bi_norm(x, *args):
	m1, s1, A1, m2,s2, A2 = args
	ret = A1*scipy.stats.norm.pdf(x, loc=m1 ,scale=s1)
	ret += A2*scipy.stats.norm.pdf(x, loc=m2 ,scale=s2)
	return ret
def tri_norm(x, *args):
	m1, s1, A1, m2,s2, A2, m3,s3, A3 = args
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
		self.maxPMerror = 5 # mas/year

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


		# initial guesses for membership
		self.RVmin = -100. #km/s
		self.RVmax = 100. #km/s
		self.RVbins = 100
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
		
		self.fitter = fitting.LevMarLSQFitter()

		# minimum membership probability to include in the CMD
		self.membershipMin = 0.01 

		self.photSigFloor = 0.01 # floor to the photometry errors for the .phot file

		# output
		self.SQLcmd = ''
		self.data = None # will be an astropy table
		self.createPlots = True # set to True to generate plots
		self.plotNameRoot = ''
		self.photOutputFileName = 'input.phot'
		self.yamlOutputFileName = 'base9.yaml'


		# dict for yaml
		# lists give [start, prior mean, prior sigma]
		self.yamlInputDict = {
			'photFile' : self.photOutputFileName,
			'outputFileBase' : 'output/base9',
			'modelDirectory' : '/projects/p31721/BASE9/base-models/',
			'msRgbModel' : 5,
			'Fe_H' : [0., 0.3, 0.3],
			'Av' : [0., 0.3, 0.3],
			'Y' : [0.29, 0.0, 0.0],
			'carbonicity' : [0.38, 0.0, 0.0],
			'logAge' : [9., np.inf, np.inf],
			'distMod' : [10., 1., 1.],
		}



	def getData(self):
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

		if (self.verbose > 2):
			print(self.data)


	def saveDataToFile(self, filename=None):
		# save the data to an ecsv file
		if (filename is None):
			filename = 'GaiaData.ecsv'

		if (self.verbose > 0):
			print(f"Saving data to file {filename} ... ")

		self.data.write(filename, overwrite=True)  

	def readDataFromFile(self, filename=None):
		# sread ave the data from an ecsv file
		if (filename is None):
			filename = 'GaiaData.ecsv'

		if (self.verbose > 0):
			print(f"Reading data from file {filename} ... ")

		self.data = ascii.read(filename)  		
	def get_minMembership(self,data):
		membership = [x for x in data if x >= self.membershipMin]
		if len(membership) < 500:
			self.membershipMin = 0.001
			membership = [x for x in data if x >= self.membershipMin]
			if len(membership) < 100:
				self.membershipMin = 0.00001
				membership = [x for x in data if x >= self.membershipMin]
				if len(membership) < 50:
					self.membershipMin = 1e-50
		if len(membership) > 1e4:
			self.membershipMin = 0.1

	def getRVMembers(self, savefig=True):
		# calculate radial-velocity memberships
		if (self.verbose > 0):
			print("Finding radial-velocity members ... ")
		
		self.data['radial_velocity'].fill_value = np.nan
		x = self.data['radial_velocity']

		
		#1D histogram
		hrv, brv = np.histogram(x.filled(), bins = self.RVbins, range=(self.RVmin, self.RVmax))

		#fit
		RVguess = brv[np.argmax(hrv)]
		if (self.RVmean is not None):
			RVguess = self.RVmean
		p_init = models.Gaussian1D(np.max(hrv), RVguess, 1) \
				+ models.Gaussian1D(5, brv[np.argmax(hrv)], 50)
		fit_p = self.fitter
		rvG1D = fit_p(p_init, brv[:-1], hrv)
		self.RVsigma = min(rvG1D.stddev_0[0],rvG1D.stddev_1[0])
		if (self.verbose > 1):
			print(rvG1D)
			print(rvG1D.parameters)

		if (self.createPlots):
			hrv, brv = np.histogram(x.filled(), bins = self.RVbins, range=(self.RVmin, self.RVmax))
			xf = np.linspace(self.RVmin, self.RVmax, self.RVbins*10)
			f, ax = plt.subplots()
			ax.step(brv[:-1],hrv, color='black')
			ax.plot(xf,rvG1D(xf), color='deeppink', lw=5)
			foo = models.Gaussian1D(*rvG1D.parameters[0:3])
			ax.plot(xf, foo(xf), color='gray')
			foo = models.Gaussian1D(*rvG1D.parameters[3:])
			ax.plot(xf, foo(xf), color='darkslateblue', ls='dashed')
			ax.set_xlabel(r'RV (km s$^{-1}$)', fontsize = 16)
			ax.set_ylabel('N', fontsize = 16)
			ax.axvline(rvG1D.parameters[1], color='tab:purple', ls='dotted')
			ax.annotate(f'RV = {rvG1D.parameters[1]:.1f} km/s', (rvG1D.parameters[1] + + 0.05*(self.RVmax - self.RVmin), 0.95*max(hrv)) )
			if (savefig):
				f.savefig(self.plotNameRoot + 'RVHist.pdf', format='PDF', bbox_inches='tight')
			
		#membership calculation
		Fc = models.Gaussian1D()
		Fc.parameters = rvG1D.parameters[:3]
		self.PRV = Fc(x)/rvG1D(x)
		self.get_minMembership(self.PRV)
		self.data['PRV'] = self.PRV

	def getPMRAmembers(self, savefig=True):
		if (self.verbose > 0):
			print("Finding prpoper_motion members ... ")
		
		x = self.data['pmra']
		#1D histogram
		hrv1, brv1 = np.histogram(x, bins = self.PMxbins, range=(self.PMxmin, self.PMxmax))
		max_peak = brv1[np.argmax(hrv1)]
		left = max([brv1[i] for i in range(len(hrv1)) if brv1[i] < max_peak and hrv1[i] <= len(x)/100])
		right = min([brv1[i] for i in range(len(hrv1)) if brv1[i] > max_peak and hrv1[i] <= len(x)/100]) #find range for fit
		hrv, brv = np.histogram(x, bins = self.PMxbins, range=(left, right))
		params = [brv[np.argmax(hrv)], 1,np.max(hrv), brv[np.argmax(hrv)], 1, np.max(hrv)]
		fitted_params,_ = scipy.optimize.curve_fit(bi_norm,brv[:-1], hrv, p0=params, method='lm',sigma=np.sqrt(hrv))
		xf = np.linspace(left, right, len(hrv))
		synth_dist = bi_norm(xf, *fitted_params)
		residuals =[hrv[i]-synth_dist[i] for i in range(len(hrv))]
		res_max_peak = brv[np.argmax(residuals)]
		res_params=[res_max_peak,.1, hrv[np.argmax(residuals)]]
		tri_params = np.append(res_params, fitted_params)
		tri_fitted_params,_ = scipy.optimize.curve_fit(tri_norm,brv[:-1], hrv, p0=tri_params, method='lm',sigma=np.sqrt(hrv))
		sigmas = [tri_fitted_params[i] for i in [1,4,7]]
		if 0.5 <= min(sigmas):
			print ('ZOOMED')
			print (len(x)/10)
			left = max([brv1[i] for i in range(len(hrv1)) if brv1[i] < max_peak and hrv1[i] <= len(x)/15])
			right = min([brv1[i] for i in range(len(hrv1)) if brv1[i] > max_peak and hrv1[i] <= len(x)/15]) #find range for fit
			hrv, brv = np.histogram(x, bins = self.PMxbins, range=(left, right))	
			params = [brv[np.argmax(hrv)], 1,np.max(hrv), brv[np.argmax(hrv)], 1, np.max(hrv)]
			fitted_params,_ = scipy.optimize.curve_fit(bi_norm,brv[:-1], hrv, p0=params, method='lm',sigma=np.sqrt(hrv))
			xf = np.linspace(left, right, len(hrv))
			synth_dist = bi_norm(xf, *fitted_params)
			residuals =[hrv[i]-synth_dist[i] for i in range(len(hrv))]
			res_max_peak = brv[np.argmax(residuals)]
			res_params=[res_max_peak,.1, hrv[np.argmax(residuals)]]
			tri_params = np.append(res_params, fitted_params)
			tri_fitted_params,_ = scipy.optimize.curve_fit(tri_norm,brv[:-1], hrv, p0=tri_params, method='lm',sigma=np.sqrt(hrv))
		sigmas = [tri_fitted_params[i] for i in [1,4,7]]
		G1_sig = [i for i in [1,4,7] if tri_fitted_params[i] == min(sigmas)][0]
		G1_params =[tri_fitted_params[G1_sig-1],tri_fitted_params[G1_sig],tri_fitted_params[G1_sig+1]]
		bi_norm_params = [tri_fitted_params[i] for i in range(len(tri_fitted_params)) if i not in [G1_sig-1,G1_sig,G1_sig+1]]
		tri_fitted_params = np.append(G1_params,bi_norm_params)
		if (self.verbose > 1):
			print(tri_fitted_params)
		if (self.createPlots):
			f, ax = plt.subplots()
			xf=np.linspace(left,right,10*self.RVbins)
			ax.step(brv[:-1],hrv, color='black')
			ax.plot(xf,tri_norm(xf, *tri_fitted_params), color='deeppink', lw=5)
			ax.plot(xf,gauss(xf, *tri_fitted_params[:3]), color='gray')
			ax.plot(xf,bi_norm(xf, *tri_fitted_params[3:]), color='darkslateblue', ls='dashed')


	def getPMDECmembers(self, savefig=True):
		if (self.verbose > 0):
			print("Finding prpoper_motion members ... ")
		
		x = self.data['pmdec']
		#1D histogram
		hrv1, brv1 = np.histogram(x, bins = self.PMxbins, range=(self.PMxmin, self.PMxmax))
		max_peak = brv1[np.argmax(hrv1)]
		left = max([brv1[i] for i in range(len(hrv1)) if brv1[i] < max_peak and hrv1[i] <= len(x)/100])
		right = min([brv1[i] for i in range(len(hrv1)) if brv1[i] > max_peak and hrv1[i] <= len(x)/100]) #find range for fit
		hrv, brv = np.histogram(x, bins = self.PMxbins, range=(left, right))
		params = [brv[np.argmax(hrv)], .1,np.max(hrv), brv[np.argmax(hrv)], 1, np.max(hrv)]
		fitted_params,_ = scipy.optimize.curve_fit(bi_norm,brv[:-1], hrv, p0=params, method='lm',sigma=np.sqrt(hrv))
		xf = np.linspace(left, right, len(hrv))
		synth_dist = bi_norm(xf, *fitted_params)
		residuals =[hrv[i]-synth_dist[i] for i in range(len(hrv))]
		res_max_peak = brv[np.argmax(residuals)]
		res_params=[res_max_peak,1, hrv[np.argmax(residuals)]]
		tri_params = np.append(res_params, fitted_params)
		tri_fitted_params,_ = scipy.optimize.curve_fit(tri_norm,brv[:-1], hrv, p0=tri_params, method='lm',sigma=np.sqrt(hrv))
		sigmas = [tri_fitted_params[i] for i in [1,4,7]]
		if 0.5 <= min(sigmas):
			print('Zoomed')
			left = max([brv1[i] for i in range(len(hrv1)) if brv1[i] < max_peak and hrv1[i] <= len(x)/15])
			right = min([brv1[i] for i in range(len(hrv1)) if brv1[i] > max_peak and hrv1[i] <= len(x)/15]) #find range for fit
			hrv, brv = np.histogram(x, bins = self.PMxbins, range=(left, right))	
			params = [brv[np.argmax(hrv)], 1,np.max(hrv), brv[np.argmax(hrv)], 1, np.max(hrv)]
			fitted_params,_ = scipy.optimize.curve_fit(bi_norm,brv[:-1], hrv, p0=params, method='lm',sigma=np.sqrt(hrv))
			xf = np.linspace(left, right, len(hrv))
			synth_dist = bi_norm(xf, *fitted_params)
			residuals =[hrv[i]-synth_dist[i] for i in range(len(hrv))]
			res_max_peak = brv[np.argmax(residuals)]
			res_params=[res_max_peak,.1, hrv[np.argmax(residuals)]]
			tri_params = np.append(res_params, fitted_params)
			tri_fitted_params,_ = scipy.optimize.curve_fit(tri_norm,brv[:-1], hrv, p0=tri_params, method='lm',sigma=np.sqrt(hrv))
		sigmas = [tri_fitted_params[i] for i in [1,4,7]]
		G1_sig = [i for i in [1,4,7] if tri_fitted_params[i] == min(sigmas)][0]
		G1_params =[tri_fitted_params[G1_sig-1],tri_fitted_params[G1_sig],tri_fitted_params[G1_sig+1]]
		bi_norm_params = [tri_fitted_params[i] for i in range(len(tri_fitted_params)) if i not in [G1_sig-1,G1_sig,G1_sig+1]]
		tri_fitted_params = np.append(G1_params,bi_norm_params)
		if (self.verbose > 1):
			print(tri_fitted_params)
		if (self.createPlots):
			f, ax = plt.subplots()
			xf=np.linspace(left,right,10*self.RVbins)
			ax.step(brv[:-1],hrv, color='black')
			ax.plot(xf,tri_norm(xf, *tri_fitted_params), color='deeppink', lw=5)
			ax.plot(xf,gauss(xf, *tri_fitted_params[:3]), color='gray')
			ax.plot(xf,bi_norm(xf, *tri_fitted_params[3:]), color='darkslateblue', ls='dashed')


	def getParallaxMembers(self, savefig=True):
		# estimate memberships based on distance (could use Bailer Jones, but this simply uses inverted parallax)
		if (self.verbose > 0):
			print("Finding parallax members ... ")
			
		x = (self.data['parallax']).to(units.parsec, equivalencies=units.parallax()).to(units.parsec).value
		
		#1D histogram
		hpa, bpa = np.histogram(x, bins = self.dbins, range=(self.dmin, self.dmax))

		#fit
		dguess = bpa[np.argmax(hpa)]
		if (self.distance is not None):
			dguess = self.distance
		p_init = models.Gaussian1D(np.max(hpa), dguess, 10)\
				+ models.Polynomial1D(degree=self.dPolyD)
		fit_p = self.fitter
		pa1D = fit_p(p_init, bpa[:-1], hpa)
		if (self.verbose > 1):
			print(pa1D)
			print(pa1D.parameters)

		if (self.createPlots):
			hpa, bpa = np.histogram(x, bins = self.dbins, range=(self.dmin, self.dmax))
			xf = np.linspace(self.dmin, self.dmax, self.dbins*10)
			f,ax = plt.subplots()
			ax.step(bpa[:-1],hpa, color='black')
			ax.plot(xf,pa1D(xf), color='deeppink', lw=5)
			foo = models.Gaussian1D(*pa1D.parameters[0:3])
			ax.plot(xf, foo(xf), color='gray')
			foo = models.Polynomial1D(degree=self.dPolyD)
			ax.plot(xf, foo.evaluate(xf,*pa1D.parameters[3:]), color='darkslateblue', ls='dashed')
			ax.set_xlabel('distance (pc)', fontsize = 16)
			ax.set_ylabel('N', fontsize = 16)
			ax.axvline(pa1D.parameters[1], color='tab:purple', ls='dotted')
			ax.annotate(f'd = {pa1D.parameters[1]:.1f} pc', (pa1D.parameters[1] + 0.05*(self.dmax - self.dmin), 0.95*max(hpa)) )
			if (savefig):
				f.savefig(self.plotNameRoot + 'dHist.pdf', format='PDF', bbox_inches='tight')
			
		#membership calculation
		Fc = models.Gaussian1D()
		Fc.parameters = pa1D.parameters[0:3]
		self.PPa = Fc(x)/pa1D(x)
		self.get_minMembership(self.PPa)
		self.data['PPa'] = self.PPa

	
	def getM35PMMembers(self, savefig=True):
		if (self.verbose > 0):
			print("finding proper-motion members ...")

		def getPMparams(x):
			#1D histogram
			hrv1, brv1 = np.histogram(x, bins = self.PMxbins, range=(self.PMxmin, self.PMxmax))
			max_peak = brv1[np.argmax(hrv1)]
			left = max([brv1[i] for i in range(len(hrv1)) if brv1[i] < max_peak and hrv1[i] <= len(x)/100])
			right = min([brv1[i] for i in range(len(hrv1)) if brv1[i] > max_peak and hrv1[i] <= len(x)/100]) #find range for fit
			hrv, brv = np.histogram(x, bins = self.PMxbins, range=(left, right))
			params = [brv[np.argmax(hrv)], .1,np.max(hrv), brv[np.argmax(hrv)], 1, np.max(hrv)]
			fitted_params,_ = scipy.optimize.curve_fit(bi_norm,brv[:-1], hrv, p0=params, method='lm',sigma=np.sqrt(hrv))
			xf = np.linspace(left, right, len(hrv))
			synth_dist = bi_norm(xf, *fitted_params)
			residuals =[hrv[i]-synth_dist[i] for i in range(len(hrv))]
			res_max_peak = brv[np.argmax(residuals)]
			res_params=[res_max_peak,1, hrv[np.argmax(residuals)]]
			tri_params = np.append(res_params, fitted_params)
			tri_fitted_params,_ = scipy.optimize.curve_fit(tri_norm,brv[:-1], hrv, p0=tri_params, method='lm',sigma=np.sqrt(hrv))
			sigmas = [tri_fitted_params[i] for i in [1,4,7]]
			if 0.5 <= min(sigmas):
				print('Zoomed')
				left = max([brv1[i] for i in range(len(hrv1)) if brv1[i] < max_peak and hrv1[i] <= len(x)/50])
				right = min([brv1[i] for i in range(len(hrv1)) if brv1[i] > max_peak and hrv1[i] <= len(x)/50]) #find range for fit
				hrv, brv = np.histogram(x, bins = self.PMxbins, range=(left, right))	
				params = [brv[np.argmax(hrv)], 1,np.max(hrv), brv[np.argmax(hrv)], 1, np.max(hrv)]
				fitted_params,_ = scipy.optimize.curve_fit(bi_norm,brv[:-1], hrv, p0=params, method='lm',sigma=np.sqrt(hrv))
				xf = np.linspace(left, right, len(hrv))
				synth_dist = bi_norm(xf, *fitted_params)
				residuals =[hrv[i]-synth_dist[i] for i in range(len(hrv))]
				res_max_peak = brv[np.argmax(residuals)]
				res_params=[res_max_peak,.1, hrv[np.argmax(residuals)]]
				tri_params = np.append(res_params, fitted_params)
				tri_fitted_params,_ = scipy.optimize.curve_fit(tri_norm,brv[:-1], hrv, p0=tri_params, method='lm',sigma=np.sqrt(hrv))
			sigmas = [tri_fitted_params[i] for i in [1,4,7]]
			G1_sig = [i for i in [1,4,7] if tri_fitted_params[i] == min(sigmas)][0]
			G1_params =[tri_fitted_params[G1_sig-1],tri_fitted_params[G1_sig],tri_fitted_params[G1_sig+1]]
			bi_norm_params = [tri_fitted_params[i] for i in range(len(tri_fitted_params)) if i not in [G1_sig-1,G1_sig,G1_sig+1]]
			tri_fitted_params = np.append(G1_params,bi_norm_params)
			synth_dist = tri_norm(xf, *tri_fitted_params)
			residuals =[hrv[i]-synth_dist[i] for i in range(len(hrv))]
			res_max_peak = brv[np.argmax(residuals)]
			res_params=[res_max_peak,1, hrv[np.argmax(residuals)]]
			quad_params = np.append(res_params, tri_fitted_params)
			quad_fitted_params,_ = scipy.optimize.curve_fit(quad_norm,brv[:-1], hrv, p0=quad_params, method='lm',sigma=np.sqrt(hrv))
			return quad_fitted_params, left, right

		
		# add in other members as a check
		members = Table(names = self.data.colnames)
		if ('PRV' in self.data.colnames):
			members = vstack([members, self.data[np.logical_and(self.data['PRV'] > self.membershipMin, ~self.data['PRV'].mask)]])
		if ('PPa' in self.data.colnames):
			members = vstack([members, self.data[self.data['PPa'] > self.membershipMin]])
		
		x = self.data['pmra']
		y = self.data['pmdec']
		x_params = getPMparams(x)
		y_params = getPMparams(y)
		PMxmin=x_params[1]
		PMxmax=x_params[2]
		PMymin=y_params[1]
		PMymax=y_params[2]

		if (self.verbose > 1):
			print(x_params,y_params)
		if (self.createPlots):
			f = plt.figure(figsize=(8, 8)) 
			gs = gridspec.GridSpec(2, 2, height_ratios = [1, 3], width_ratios = [3, 1]) 
			ax1 = plt.subplot(gs[0])
			ax2 = plt.subplot(gs[2])
			ax3 = plt.subplot(gs[3])

			#histograms
			hx1D, x1D = np.histogram(x,bins=self.PMxbins,range=(PMxmin,PMxmax))
			xf = np.linspace(PMxmin,PMxmax,self.PMxbins)
			ax1.step(x1D[:-1], hx1D, color='black')
			ax1.plot(xf, quad_norm(xf, *x_params[0]), color='deeppink', lw=3)
			ax1.plot(xf, gauss(xf, *x_params[0][:3]), color='gray')
			ax1.plot(xf, tri_norm(xf, *x_params[0][3:]), color='darkslateblue', ls='dashed')
			ax1.axvline(x_params[0][0],color='tab:purple', ls='dotted')
			ax1.annotate(r'$\mu_\alpha$ =' + f'{x_params[0][0]:.1f}' + r'mas yr$^{-1}$', (x_params[0][0] + 0.05*(x_params[2] - x_params[1]), 0.95*max(hx1D)) )
			ax1.set_ylim(-.05*max(hx1D),max(hx1D)+.1*max(hx1D))

			hy1D, y1D = np.histogram(y,bins=self.PMybins,range=(PMymin,PMymax))
			yf = np.linspace(PMymin,PMymax,200)
			ax3.step(hy1D, y1D[:-1], color='black')
			ax3.plot(quad_norm(yf, *y_params[0]),yf, color='deeppink', lw=3)
			ax3.plot(gauss(yf, *y_params[0][:3]),yf, color='gray')
			ax3.plot(tri_norm(yf, *y_params[0][3:]),yf, color='darkslateblue', ls='dashed')
			ax3.axhline(y_params[0][0],color='tab:purple', ls='dotted')
			ax3.annotate(r'$\mu_\alpha$ =' + f'{y_params[0][0]:.1f}' + r'mas yr$^{-1}$',(0.95*max(hy1D), y_params[0][0] + 0.05*(y_params[2] - y_params[1])), rotation=90)
			ax3.set_xlim(-.05*max(hy1D),max(hy1D)+.1*max(hy1D))

			ax2.scatter(x,y,color='darkgray',marker='.')
			ax2.scatter(members['pmra'], members['pmdec'], color='cyan', marker='.')

			ax1.set_xlim(PMxmin, PMxmax)
			ax2.set_xlim(PMxmin, PMxmax)
			ax2.set_ylim(PMymin, PMymax)
			ax3.set_ylim(PMymin, PMymax)
			ax2.set_xlabel(r'$\mu_\alpha$ (mas yr$^{-1}$)', fontsize=16)
			ax2.set_ylabel(r'$\mu_\delta$ (mas yr$^{-1}$)', fontsize=16)
			plt.setp(ax1.get_yticklabels()[0], visible=False)
			plt.setp(ax1.get_xticklabels(), visible=False)
			plt.setp(ax3.get_yticklabels(), visible=False)
			plt.setp(ax3.get_xticklabels()[0], visible=False)
			f.subplots_adjust(hspace=0., wspace=0.)
			if (savefig):
				f.savefig(self.plotNameRoot + 'PMHist.pdf', format='PDF', bbox_inches='tight')

		#membership calculation
		RA_mems = gauss(x, *x_params[0][:3])/quad_norm(x, *x_params[0])
		DEC_mems = gauss(y, *y_params[0][:3])/quad_norm(y, *y_params[0])
		self.PPM = np.array(RA_mems*DEC_mems)
		self.data['PPM'] = self.PPM

	def getPMMembers(self, savefig=True):
		if (self.verbose > 0):
			print("finding proper-motion members ...")

		def getPMparams(x):
		#1D histogram
			hrv1, brv1 = np.histogram(x, bins = self.PMxbins, range=(self.PMxmin, self.PMxmax))
			max_peak = brv1[np.argmax(hrv1)]
			left = max([brv1[i] for i in range(len(hrv1)) if brv1[i] < max_peak and hrv1[i] <= len(x)/100])
			right = min([brv1[i] for i in range(len(hrv1)) if brv1[i] > max_peak and hrv1[i] <= len(x)/100]) #find range for fit
			hrv, brv = np.histogram(x, bins = self.PMxbins, range=(left, right))
			params = [brv[np.argmax(hrv)], .1,np.max(hrv), brv[np.argmax(hrv)], 1, np.max(hrv)]
			fitted_params,_ = scipy.optimize.curve_fit(bi_norm,brv[:-1], hrv, p0=params, method='lm',sigma=np.sqrt(hrv))
			xf = np.linspace(left, right, len(hrv))
			synth_dist = bi_norm(xf, *fitted_params)
			residuals =[hrv[i]-synth_dist[i] for i in range(len(hrv))]
			res_max_peak = brv[np.argmax(residuals)]
			res_params=[res_max_peak,1, hrv[np.argmax(residuals)]]
			tri_params = np.append(res_params, fitted_params)
			tri_fitted_params,_ = scipy.optimize.curve_fit(tri_norm,brv[:-1], hrv, p0=tri_params, method='lm',sigma=np.sqrt(hrv))
			sigmas = [tri_fitted_params[i] for i in [1,4,7]]
			if 0.5 <= min(sigmas):
				print ('ZOOMED')
				left = max([brv1[i] for i in range(len(hrv1)) if brv1[i] < max_peak and hrv1[i] <= len(x)/15])
				right = min([brv1[i] for i in range(len(hrv1)) if brv1[i] > max_peak and hrv1[i] <= len(x)/15]) #find range for fit
				hrv, brv = np.histogram(x, bins = self.PMxbins, range=(left, right))	
				params = [brv[np.argmax(hrv)], 1,np.max(hrv), brv[np.argmax(hrv)], 1, np.max(hrv)]
				fitted_params,_ = scipy.optimize.curve_fit(bi_norm,brv[:-1], hrv, p0=params, method='lm',sigma=np.sqrt(hrv))
				xf = np.linspace(left, right, len(hrv))
				synth_dist = bi_norm(xf, *fitted_params)
				residuals =[hrv[i]-synth_dist[i] for i in range(len(hrv))]
				res_max_peak = brv[np.argmax(residuals)]
				res_params=[res_max_peak,.1, hrv[np.argmax(residuals)]]
				tri_params = np.append(res_params, fitted_params)
				tri_fitted_params,_ = scipy.optimize.curve_fit(tri_norm,brv[:-1], hrv, p0=tri_params, method='lm',sigma=np.sqrt(hrv))
			sigmas = [tri_fitted_params[i] for i in [1,4,7]]
			G1_sig = [i for i in [1,4,7] if tri_fitted_params[i] == min(sigmas)][0]
			G1_params =[tri_fitted_params[G1_sig-1],tri_fitted_params[G1_sig],tri_fitted_params[G1_sig+1]]
			bi_norm_params = [tri_fitted_params[i] for i in range(len(tri_fitted_params)) if i not in [G1_sig-1,G1_sig,G1_sig+1]]
			tri_fitted_params = np.append(G1_params,bi_norm_params)
			return tri_fitted_params,left,right

		
		# add in other members as a check
		members = Table(names = self.data.colnames)
		if ('PRV' in self.data.colnames):
			members = vstack([members, self.data[np.logical_and(self.data['PRV'] > self.membershipMin, ~self.data['PRV'].mask)]])
		if ('PPa' in self.data.colnames):
			members = vstack([members, self.data[self.data['PPa'] > self.membershipMin]])
		
		x = self.data['pmra']
		y = self.data['pmdec']
		x_params = getPMparams(x)
		y_params = getPMparams(y)
		PMxmin=x_params[1]
		PMxmax=x_params[2]
		PMymin=y_params[1]
		PMymax=y_params[2]

		if (self.verbose > 1):
			print(x_params,y_params)
		if (self.createPlots):
			f = plt.figure(figsize=(8, 8)) 
			gs = gridspec.GridSpec(2, 2, height_ratios = [1, 3], width_ratios = [3, 1]) 
			ax1 = plt.subplot(gs[0])
			ax2 = plt.subplot(gs[2])
			ax3 = plt.subplot(gs[3])

			#histograms
			hx1D, x1D = np.histogram(x,bins=self.PMxbins,range=(PMxmin,PMxmax))
			xf = np.linspace(PMxmin,PMxmax,self.PMxbins)
			ax1.step(x1D[:-1], hx1D, color='black')
			ax1.plot(xf, tri_norm(xf, *x_params[0]), color='deeppink', lw=3)
			ax1.plot(xf, gauss(xf, *x_params[0][:3]), color='gray')
			ax1.plot(xf, bi_norm(xf, *x_params[0][3:]), color='darkslateblue', ls='dashed')
			ax1.axvline(x_params[0][0],color='tab:purple', ls='dotted')
			ax1.annotate(r'$\mu_\alpha$ =' + f'{x_params[0][0]:.1f}' + r'mas yr$^{-1}$', (x_params[0][0] + 0.05*(x_params[2] - x_params[1]), 0.95*max(hx1D)) )
			ax1.set_ylim(-.05*max(hx1D),max(hx1D)+.1*max(hx1D))

			hy1D, y1D = np.histogram(y,bins=self.PMybins,range=(PMymin,PMymax))
			yf = np.linspace(PMymin,PMymax,200)
			ax3.step(hy1D, y1D[:-1], color='black')
			ax3.plot(tri_norm(yf, *y_params[0]),yf, color='deeppink', lw=3)
			ax3.plot(gauss(yf, *y_params[0][:3]),yf, color='gray')
			ax3.plot(bi_norm(yf, *y_params[0][3:]),yf, color='darkslateblue', ls='dashed')
			ax3.axhline(y_params[0][0],color='tab:purple', ls='dotted')
			ax3.annotate(r'$\mu_\alpha$ =' + f'{y_params[0][0]:.1f}' + r'mas yr$^{-1}$',(0.95*max(hy1D), y_params[0][0] + 0.05*(y_params[2] - y_params[1])), rotation=90)
			ax3.set_xlim(-.05*max(hy1D),max(hy1D)+.1*max(hy1D))

			ax2.scatter(x,y,color='darkgray',marker='.')
			ax2.scatter(members['pmra'], members['pmdec'], color='cyan', marker='.')

			ax1.set_xlim(PMxmin, PMxmax)
			ax2.set_xlim(PMxmin, PMxmax)
			ax2.set_ylim(PMymin, PMymax)
			ax3.set_ylim(PMymin, PMymax)
			ax2.set_xlabel(r'$\mu_\alpha$ (mas yr$^{-1}$)', fontsize=16)
			ax2.set_ylabel(r'$\mu_\delta$ (mas yr$^{-1}$)', fontsize=16)
			plt.setp(ax1.get_yticklabels()[0], visible=False)
			plt.setp(ax1.get_xticklabels(), visible=False)
			plt.setp(ax3.get_yticklabels(), visible=False)
			plt.setp(ax3.get_xticklabels()[0], visible=False)
			f.subplots_adjust(hspace=0., wspace=0.)
			if (savefig):
				f.savefig(self.plotNameRoot + 'PMHist.pdf', format='PDF', bbox_inches='tight')

		#membership calculation
		RA_mems = gauss(x, *x_params[0][:3])/tri_norm(x, *x_params[0])
		DEC_mems = gauss(y, *y_params[0][:3])/tri_norm(y, *y_params[0])
		self.PPM = np.array(RA_mems*DEC_mems)
		self.data['PPM'] = self.PPM

	def combineMemberships(self):
		if (self.verbose > 0):
			print("combining memberships ...")
		# I'm not sure the best way to combine these
		# We probably want to multiple them together, but if there is no membership (e.g., in RV), then we still keep the star
		self.data['PRV'].fill_value = 1.
		#self.data['PPa'].fill_value = 1.  # it appears that this is not a masked column
		self.data['PPM'].fill_value = 1.
		self.data['membership'] = np.nan_to_num(self.data['PRV'].filled(), nan=1)*\
								  np.nan_to_num(self.data['PPa'], nan=1)*\
								  np.nan_to_num(self.data['PPM'], nan=1)

	def plotCMD(self, data=None, x1='g_mean_psf_mag', x2='i_mean_psf_mag', y='g_mean_psf_mag', m='membership', savefig=True):
		if (self.verbose > 0):
			print("plotting CMD ...")

		if (data is None):
			data = self.data

		# I could specify the columns to use
		f, ax = plt.subplots(figsize=(5,8))
		ax.plot(data[x1] - data[x2], data[y],'.', color='lightgray')

		#members
		mask = (data[m] > self.membershipMin)
		self.get_minMembership(data[mask][y])
		ax.plot(data[mask][x1] - data[mask][x2], data[mask][y],'.', color='deeppink')
		ax.set_ylim(max(data[mask][y]), min(data[mask][y]))
		#ax.set_xlim(-1, 5)
		ax.set_xlabel(x1+'-'+x2, fontsize=16)
		ax.set_ylabel(y, fontsize=16)
		if (savefig):
			f.savefig(self.plotNameRoot + 'CMD.pdf', format='PDF', bbox_inches='tight')

	def generatePhotFile(self):
		if (self.verbose > 0):
			print("generating phot file ...")

		# create a *.phot file for input to BASE-9
		# would be nice if this was more general and could handle any set of photometry

		# take only those that pass the membership threshold
		mask = (self.data['membership'] > self.membershipMin) 
		members = self.data[mask]

		# sort the data by distance from the (user defined) center and also by magnitude to generate IDs
		center = SkyCoord(self.RA*units.degree, self.Dec*units.degree, frame='icrs')
		members['coord'] = SkyCoord(members['ra'], members['dec'], frame='icrs') 
		members['rCenter'] = center.separation(members['coord'])
		# create 10 annuli to help with IDs?
		# members['annulus'] = (np.around(members['rCenter'].to(units.deg).value, decimals = 1)*10 + 1).astype(int)
		# add indices for the radii from the center and g mag for IDs
		members.sort(['rCenter'])
		members['rRank'] = np.arange(0,len(members)) + 1
		members.sort(['phot_g_mean_mag'])
		members['gRank'] = np.arange(0,len(members)) + 1
		epoch = 1
		zfillN = int(np.ceil(np.log10(len(members))))
		members['id'] = [str(epoch) + str(r).zfill(zfillN) + str(g).zfill(zfillN) for (r,g) in zip(members['rRank'].value, members['gRank'].value) ]

		# include only the columns we need in the output table.
		# Currently I am not including Gaia photometry
		# If we want to include Gaia photometry, we need to include errors.  
		# Maybe we can use "typical errors" from here: https://gea.esac.esa.int/archive/documentation/GEDR3/index.html
		# add the extra columns for BASE-9
		members['mass1'] = np.zeros(len(members)) #if we know masses, these could be added
		members['massRatio'] = np.zeros(len(members)) #if we know mass ratios, these could be added
		members['stage1'] = np.zeros(len(members)) + 1 #set to 1 for MS and giant stars (use 2(?) for WDs)
		members['useDBI'] = np.zeros(len(members)) + 1 #set to 1 to use during burn-in.  May want to improve to remove anomalous stars
		out = members[['id', 
					   'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
					   'g_mean_psf_mag', 'r_mean_psf_mag', 'i_mean_psf_mag', 'z_mean_psf_mag', 'y_mean_psf_mag',
					   'j_m', 'h_m', 'ks_m',
					   'phot_g_mean_mag_error', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error',
					   'g_mean_psf_mag_error', 'r_mean_psf_mag_error', 'i_mean_psf_mag_error', 'z_mean_psf_mag_error', 'y_mean_psf_mag_error',
					   'j_msigcom', 'h_msigcom', 'ks_msigcom',
					   'mass1', 'massRatio', 'stage1','membership','useDBI'
					   ]]
		# rename columns
		out.rename_column('phot_g_mean_mag', 'G')
		out.rename_column('phot_bp_mean_mag', 'G_BP') 
		out.rename_column('phot_rp_mean_mag', 'G_RP')
		out.rename_column('g_mean_psf_mag', 'g_ps')
		out.rename_column('r_mean_psf_mag', 'r_ps')
		out.rename_column('i_mean_psf_mag', 'i_ps')
		out.rename_column('z_mean_psf_mag', 'z_ps')
		out.rename_column('y_mean_psf_mag', 'y_ps')
		out.rename_column('phot_g_mean_mag_error', 'sigG')
		out.rename_column('phot_bp_mean_mag_error', 'sigG_BP') 
		out.rename_column('phot_rp_mean_mag_error', 'sigG_RP')
		out.rename_column('g_mean_psf_mag_error', 'sigg_ps')
		out.rename_column('r_mean_psf_mag_error', 'sigr_ps')
		out.rename_column('i_mean_psf_mag_error', 'sigi_ps')
		out.rename_column('z_mean_psf_mag_error', 'sigz_ps')
		out.rename_column('y_mean_psf_mag_error', 'sigy_ps')
		out.rename_column('j_m', 'J_2M')
		out.rename_column('h_m', 'H_2M')
		out.rename_column('ks_m', 'Ks_2M')
		out.rename_column('j_msigcom', 'sigJ_2M')
		out.rename_column('h_msigcom', 'sigH_2M')
		out.rename_column('ks_msigcom', 'sigKs_2M')
		out.rename_column('membership', 'CMprior')

		# impose a floor to phot error to be safe
		for c in ['sigG', 'sigG_BP', 'sigG_RP', 'sigg_ps', 'sigr_ps', 'sigi_ps', 'sigz_ps', 'sigy_ps', 'sigJ_2M', 'sigH_2M', 'sigKs_2M']:
			out[c][(out[c] < self.photSigFloor)] = self.photSigFloor

		# replace any nan or mask values with 99.9 for mag and -9.9 for sig
		for c in ['G', 'G_BP', 'G_RP', 'g_ps', 'r_ps', 'i_ps', 'z_ps', 'y_ps', 'J_2M', 'H_2M', 'Ks_2M']:
			out[c].fill_value = 99.9
			out[c] = out[c].filled()
		for c in ['sigG', 'sigG_BP', 'sigG_RP', 'sigg_ps', 'sigr_ps', 'sigi_ps', 'sigz_ps', 'sigy_ps', 'sigJ_2M', 'sigH_2M', 'sigKs_2M']:
			out[c].fill_value = -9.9
			out[c] = out[c].filled()

		# expose this so it can be used elsewhere
		self.members = members

		# write the phot file
		self.dumpPhotFile(out)


	def dumpPhotFile(self, out, filename=None):
		if (filename is None):
			filename = self.photOutputFileName

		idint = list(map(int, out['id']))
		zfillN = int(np.ceil(np.log10(max(idint)))) 

		# write to file with proper formatting
		# fdec = np.abs(np.log10(self.photSigFloor)).astype(int)
		# ffmt = '%-' + str(fdec + 3) + '.' + str(fdec) + 'f'
		ffmt = '%-7.4f'
		with open(filename, 'w', newline='\n') as f:
			ascii.write(out, delimiter=' ', output=f, format = 'basic',
				formats = {'id': '%' + str(2*zfillN + 1) + 's', 
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

		yamlOutput['general']['cluster']['priors']['sigmas']['Fe_H'] = float(self.yamlInputDict['Fe_H'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['Av'] = float(self.yamlInputDict['Av'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['Y'] = float(self.yamlInputDict['Y'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['carbonicity'] = float(self.yamlInputDict['carbonicity'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['logAge'] = float(self.yamlInputDict['logAge'][2])
		yamlOutput['general']['cluster']['priors']['sigmas']['distMod'] = float(self.yamlInputDict['distMod'][2])


		# I hope this doesn't need to be sorted in the same order as the original
		# This outputs in alphabetical order
		with open(self.yamlOutputFileName, 'w') as file:
			yaml.dump(yamlOutput, file, indent = 4)

	def createInteractive(self, mag = 'G', color1 = 'G_BP', color2 = 'G_RP', xrng = [0.5,2], yrng = [20,10]):
		# NOTE: currently this code requires a column in the data labelled as 'membership'

		# create the initial figure
		TOOLS = "box_zoom, reset, lasso_select, box_select"
		p = figure(title = "",
			tools = TOOLS, width = 500, height = 700,
			x_range = xrng, y_range = yrng)

		# set all useDBI = 0 to start
		self.data['useDBI'] = [0]*len(self.data)

		mask = (self.data['membership'] > self.membershipMin) 
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

		# add filled polygon to hold the region within the selection zone for the plot

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

		###########################
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
			mask = (self.data['membership'] > self.membershipMin)
			sourcePhot.data = dict(x = self.data[mask][color1] - self.data[mask][color2], y = self.data[mask][mag], index = self.data[mask]['index'])
		resetButton.on_click(resetCallback)

		# text box to define the output file
		# outfile = TextInput(value = datafile + '.new', title = "Output File Name:")

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
				mask = (self.data['membership'] > self.membershipMin)
				sourcePhot.data = dict(x = self.data[mask][color1] - self.data[mask][color2], y = self.data[mask][mag], index = self.data[mask]['index'])
				# reset
				sourcePhot.selected.indices = []

		deleteButton.on_click(deleteCallback)

		###########################
		# layout
		# plot on the left, buttons on the right
		buttons = column(
			slider, 
			Div(text='<div style="height: 15px;"></div>'),
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


	def runAll(self, clusterName):
		if clusterName == 'M_35':
			self.radius = 2.*self.radius/3.
			self.getData()
			self.getRVMembers()
			self.getParallaxMembers()
			self.getM35PMMembers(clusterName)
		else:
			self.getData()
			self.getRVMembers()
			self.getParallaxMembers()
			self.getPMMembers(clusterName)
		self.combineMemberships()
		self.plotCMD(y = 'phot_g_mean_mag', x1 = 'phot_bp_mean_mag', x2 = 'phot_rp_mean_mag')

		if (self.verbose > 0):
			print("done.")
