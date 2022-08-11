# Python code to download Gaia data at a given location

from astroquery.gaia import Gaia
from astropy.modeling import models, fitting
from astropy.table import Table, Column
import astropy.units as units
from astropy.coordinates import SkyCoord
from astropy.io import ascii

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mplColors
import matplotlib.cm as cm

import yaml

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
		self.dmax = 3000. #parsecs
		self.dbins = 200
		self.dPolyD = 6 #degrees for polynomial fit for distance distribution
		self.PMxmin = -200 #mas/yr
		self.PMxmax = 200 #mas/yr
		self.PMxbins = 400
		self.PMymin = -200 #mas/yr
		self.PMymax = 200 #mas/yr
		self.PMybins = 400  
		self.RVmean = None #could explicitly set the mean cluster RV for the initial guess
		self.distance = None #could explicitly set the mean cluster distance for the initial guess
		self.PMmean = [None, None] #could explicitly set the mean cluster PM for the initial guess
		
		self.fitter = fitting.LevMarLSQFitter()

		# minimum membership probability to include in the CMD
		self.membershipMin = 0.0 

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
			'modelDirectory' : 'base-models/',
			'msRgbModel' : 5,
			'Fe_H' : [0., 0., 0.3],
			'Av' : [0., 0., 0.3],
			'Y' : [0.29, 0.29, 0.0],
			'carbonicity' : [0.38, 0.38, 0.0],
			'logAge' : [9., 9., np.inf],
			'distMod' : [10., 10., 1.],
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

	def getRVMembers(self, savefig=True):
		# calculate radial-velocity memberships
		if (self.verbose > 0):
			print("Finding radial-velocity members ... ")
		
		x = self.data['radial_velocity']
		
		#1D histogram
		hrv, brv = np.histogram(x, bins = self.RVbins, range=(self.RVmin, self.RVmax))

		#fit
		RVguess = brv[np.argmax(hrv)]
		if (self.RVmean is not None):
			RVguess = self.RVmean
		p_init = models.Gaussian1D(np.max(hrv), RVguess, 1) \
				+ models.Gaussian1D(5, brv[np.argmax(hrv)], 50)
		fit_p = self.fitter
		rvG1D = fit_p(p_init, brv[:-1], hrv)
		if (self.verbose > 1):
			print(rvG1D)
			print(rvG1D.parameters)

		if (self.createPlots):
			hrv, brv = np.histogram(x, bins = self.RVbins, range=(self.RVmin, self.RVmax))
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
		Fc.parameters = rvG1D.parameters[0:3]
		self.PRV = Fc(x)/rvG1D(x)
		self.data['PRV'] = self.PRV

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
		self.data['PPa'] = self.PPa


	def getPMMembers2Step(self, savefig=True):

		if (self.verbose > 0):
			print("finding proper-motion members with two steps...")

		x = self.data['pmra']#*np.cos(self.data['dec']*np.pi/180.)
		y = self.data['pmdec']

		#1D histograms (use the members here)          
		pmRAbins = np.linspace(self.PMxmin, self.PMxmax, self.PMxbins)
		pmDecbins = np.linspace(self.PMymin, self.PMymax, self.PMybins)
		hx1D, x1D = np.histogram(x, bins=pmRAbins)
		hy1D, y1D = np.histogram(y, bins=pmDecbins)

		#2D histogram
		h2D, x2D, y2D = np.histogram2d(x, y, bins=[self.PMxbins, self.PMybins], \
									   range=[[self.PMxmin, self.PMxmax], [self.PMymin, self.PMymax]])

		# for the fitter
		fit_p = self.fitter
		xf, yf = np.meshgrid(x2D[:-1], y2D[:-1], indexing='ij')

		##########
		# Field
		PMxguess = x1D[np.argmax(hx1D)]
		PMyguess = y1D[np.argmax(hy1D)]

		p_init = models.Gaussian2D(np.max(h2D.flatten()), PMxguess, PMyguess, 5, 5)
		pmG2D_field = fit_p(p_init, xf, yf, h2D)
		if (self.verbose > 1):
			print("Field fit:")
			print(pmG2D_field)
			print(pmG2D_field.parameters)
		##########

		##########
		# Subtracted data
		# get the fit values in each bin
		field_values = pmG2D_field(xf, yf).T
		subtracted_data = h2D - pmG2D_field(xf, yf)
		##########


		#########
		# Cluster as subtracted data
		if (self.PMmean[0] is not None):
			PMxguess = self.PMmean[0]
		if (self.PMmean[1] is not None):
			PMyguess = self.PMmean[1]

		p_init = models.Gaussian2D(np.max(subtracted_data.flatten()), PMxguess, PMyguess, 1, 1)

		pmG2D_cluster = fit_p(p_init, xf, yf, subtracted_data)
		if (self.verbose > 1):
			print("Cluster fit:")
			print(pmG2D_cluster)
			print(pmG2D_cluster.parameters)
		#########

		if (self.createPlots):
			# make the plots
			f = plt.figure(figsize=(16, 8)) 
			gs0 = gridspec.GridSpec(1, 2, figure=f)

			gs_field = gridspec.GridSpecFromSubplotSpec(2, 2, height_ratios = [1, 3], width_ratios = [3, 1], subplot_spec=gs0[0], hspace=0, wspace=0)
			ax1_field = plt.subplot(gs_field[0])
			ax2_field = plt.subplot(gs_field[2])
			ax3_field = plt.subplot(gs_field[3])

			###### for the field
			#histograms
			hx1D, x1D = np.histogram(x, bins=pmRAbins)
			ax1_field.step(x1D[:-1], hx1D, color='black')
			ax1_field.plot(x2D[:-1], np.sum(pmG2D_field(xf, yf), axis=1), color='deeppink', lw=5)
			foo = models.Gaussian2D(*pmG2D_field.parameters[0:6])
			ax1_field.plot(x2D[:-1], np.sum(foo(xf, yf), axis=1), color='gray')
			foo = models.Gaussian2D(*pmG2D_field.parameters[6:6])
			ax1_field.plot(x2D[:-1], np.sum(foo(xf, yf), axis=1), color='darkslateblue', ls='dashed')
			ax1_field.axvline(pmG2D_field.parameters[1], color='tab:purple', ls='dotted')
			ax1_field.annotate(r'$\mu_\alpha$ =' + f'{pmG2D_field.parameters[1]:.1f}' + r'mas yr$^{-1}$', (pmG2D_field.parameters[1] + 0.05*(self.PMxmax - self.PMxmin), 0.95*max(hx1D)) )

			hy1D, y1D = np.histogram(y, bins=pmDecbins)
			ax3_field.step(hy1D, y1D[:-1], color='black')
			ax3_field.plot(np.sum(pmG2D_field(xf, yf), axis=0), y2D[:-1], color='deeppink', lw=5)
			foo = models.Gaussian2D(*pmG2D_field.parameters[0:6])
			ax3_field.plot(np.sum(foo(xf, yf), axis=0), y2D[:-1], color='gray')

			#heatmap
			h2D, x2D, y2D, im = ax2_field.hist2d(x, y, bins=[self.PMxbins, self.PMybins],\
										   range=[[self.PMxmin, self.PMxmax], [self.PMymin, self.PMymax]], \
										   norm = mplColors.LogNorm(), cmap = cm.Greys)
			ax2_field.contourf(x2D[:-1], y2D[:-1], pmG2D_field(xf, yf).T, cmap=cm.RdPu, bins = 20, \
						 norm=mplColors.LogNorm(), alpha = 0.3)

			ax1_field.set_xlim(self.PMxmin, self.PMxmax)
			ax2_field.set_xlim(self.PMxmin, self.PMxmax)
			ax2_field.set_ylim(self.PMymin, self.PMymax)
			ax3_field.set_ylim(self.PMymin, self.PMymax)
			ax2_field.set_xlabel(r'$\mu_\alpha$ (mas yr$^{-1}$)', fontsize=16)
			ax2_field.set_ylabel(r'$\mu_\delta$ (mas yr$^{-1}$)', fontsize=16)
			plt.setp(ax1_field.get_yticklabels()[0], visible=False)
			plt.setp(ax1_field.get_xticklabels(), visible=False)
			plt.setp(ax3_field.get_yticklabels(), visible=False)
			plt.setp(ax3_field.get_xticklabels()[0], visible=False)


			###### for the cluster
			gs_cluster = gridspec.GridSpecFromSubplotSpec(2, 2, height_ratios = [1, 3], width_ratios = [3, 1], subplot_spec=gs0[1], hspace=0, wspace=0)
			ax1_cluster = plt.subplot(gs_cluster[0])
			ax2_cluster = plt.subplot(gs_cluster[2])
			ax3_cluster = plt.subplot(gs_cluster[3])

			#histograms
			ax1_cluster.plot(x2D[:-1], np.sum(subtracted_data, axis=1), color='black', lw=1)
			ax1_cluster.plot(x2D[:-1], np.sum(pmG2D_cluster(xf, yf), axis=1), color='deeppink', lw=5)
			ax3_cluster.plot(np.sum(subtracted_data, axis=0), y2D[:-1], color='black', lw=1)
			ax3_cluster.plot(np.sum(pmG2D_cluster(xf, yf), axis=0), y2D[:-1], color='deeppink', lw=5)

			#heatmap
			ax2_cluster.contourf(x2D[:-1], y2D[:-1], subtracted_data.T, cmap=cm.Greys, norm=mplColors.LogNorm(), alpha = 0.3)
			ax2_cluster.contourf(x2D[:-1], y2D[:-1], pmG2D_cluster(xf, yf).T, cmap=cm.RdPu, bins = 20, \
						 norm=mplColors.LogNorm(), alpha = 0.3)

			ax1_cluster.set_xlim(self.PMxmin, self.PMxmax)
			ax2_cluster.set_xlim(self.PMxmin, self.PMxmax)
			ax2_cluster.set_ylim(self.PMymin, self.PMymax)
			ax3_cluster.set_ylim(self.PMymin, self.PMymax)

			ax2_cluster.set_xlabel(r'$\mu_\alpha$ (mas yr$^{-1}$)', fontsize=16)
			ax2_cluster.set_ylabel(r'$\mu_\delta$ (mas yr$^{-1}$)', fontsize=16)
			plt.setp(ax1_cluster.get_yticklabels()[0], visible=False)
			plt.setp(ax1_cluster.get_xticklabels(), visible=False)
			plt.setp(ax3_cluster.get_yticklabels(), visible=False)
			plt.setp(ax3_cluster.get_xticklabels()[0], visible=False)
			if (savefig):
				f.savefig(self.plotNameRoot + 'PMHist2Step.pdf', format='PDF', bbox_inches='tight')										
		#membership calculation
		self.PPM = pmG2D_cluster(x,y)/(pmG2D_cluster(x,y) + pmG2D_field(x,y))
		self.data['PPM'] = self.PPM						
											
	def getPMMembers(self, savefig=True):
		if (self.verbose > 0):
			print("finding proper-motion members ...")
		
		x = self.data['pmra']#*np.cos(self.data['dec']*np.pi/180.)
		y = self.data['pmdec']
		
		#1D histograms (use the members here)          
		pmRAbins = np.linspace(self.PMxmin, self.PMxmax, self.PMxbins)
		pmDecbins = np.linspace(self.PMymin, self.PMymax, self.PMybins)
		hx1D, x1D = np.histogram(x, bins=pmRAbins)
		hy1D, y1D = np.histogram(y, bins=pmDecbins)

		#2D histogram
		h2D, x2D, y2D = np.histogram2d(x, y, bins=[self.PMxbins, self.PMybins], \
									   range=[[self.PMxmin, self.PMxmax], [self.PMymin, self.PMymax]])
		
		#fit
		PMxguess = x1D[np.argmax(hx1D)]
		PMyguess = y1D[np.argmax(hy1D)]
		if (self.PMmean[0] is not None):
			PMxguess = self.PMmean[0]
		if (self.PMmean[1] is not None):
			PMyguess = self.PMmean[1]
		p_init = models.Gaussian2D(np.max(h2D.flatten()), PMxguess, PMyguess, 1, 1)\
				+ models.Gaussian2D(np.max(h2D.flatten()), 0, 0, 5, 5)
		# p_init = models.Gaussian2D(np.max(h2D.flatten()), PMxguess, PMyguess, 1, 1)\
		# 		+ models.Polynomial2D(degree = 2)
		fit_p = self.fitter
		xf, yf = np.meshgrid(x2D[:-1], y2D[:-1], indexing='ij')
		pmG2D = fit_p(p_init, xf, yf, h2D)
		if (self.verbose > 1):
			print(pmG2D)
			print(pmG2D.parameters)
			
		if (self.createPlots):
			f = plt.figure(figsize=(8, 8)) 
			gs = gridspec.GridSpec(2, 2, height_ratios = [1, 3], width_ratios = [3, 1]) 
			ax1 = plt.subplot(gs[0])
			ax2 = plt.subplot(gs[2])
			ax3 = plt.subplot(gs[3])

			#histograms
			hx1D, x1D = np.histogram(x, bins=pmRAbins)
			ax1.step(x1D[:-1], hx1D, color='black')
			ax1.plot(x2D[:-1], np.sum(pmG2D(xf, yf), axis=1), color='deeppink', lw=5)
			#foo = models.Gaussian2D(*pmG2D.parameters[0:6])
			foo = models.Gaussian2D(pmG2D.amplitude_0,pmG2D.x_mean_0,pmG2D.y_mean_0,pmG2D.x_stddev_0, pmG2D.y_stddev_0)
			ax1.plot(x2D[:-1], np.sum(foo(xf, yf), axis=1), color='gray')
			foo = models.Gaussian2D(pmG2D.amplitude_1,pmG2D.x_mean_1,pmG2D.y_mean_1,pmG2D.x_stddev_1, pmG2D.y_stddev_1)
			ax1.plot(x2D[:-1], np.sum(foo(xf, yf), axis=1), color='darkslateblue', ls='dashed')
			ax1.axvline(pmG2D.parameters[1], color='tab:purple', ls='dotted')
			ax1.annotate(r'$\mu_\alpha$ =' + f'{pmG2D.parameters[1]:.1f}' + r'mas yr$^{-1}$', (pmG2D.parameters[1] + 0.05*(self.PMxmax - self.PMxmin), 0.95*max(hx1D)) )

			hy1D, y1D = np.histogram(y, bins=pmDecbins)
			ax3.step(hy1D, y1D[:-1], color='black')
			ax3.plot(np.sum(pmG2D(xf, yf), axis=0), y2D[:-1], color='deeppink', lw=5)
			foo = models.Gaussian2D(pmG2D.amplitude_0,pmG2D.x_mean_0,pmG2D.y_mean_0,pmG2D.x_stddev_0, pmG2D.y_stddev_0)
			ax3.plot(np.sum(foo(xf, yf), axis=0),y2D[:-1], color='gray')
			foo = models.Gaussian2D(pmG2D.amplitude_1,pmG2D.x_mean_1,pmG2D.y_mean_1,pmG2D.x_stddev_1, pmG2D.y_stddev_1)
			ax3.plot(np.sum(foo(xf, yf), axis=0), y2D[:-1], color='darkslateblue', ls='dashed')
			ax3.axhline(pmG2D.parameters[2], color='tab:purple', ls='dotted')
			ax3.annotate(r'$\mu_\delta$ =' + f'{pmG2D.parameters[2]:.1f}' + r'mas yr$^{-1}$', (0.95*max(hy1D), pmG2D.parameters[2] + 0.05*(self.PMymax - self.PMymin)), rotation=90)

			#heatmap
			h2D, x2D, y2D, im = ax2.hist2d(x, y, bins=[self.PMxbins, self.PMybins],\
										   range=[[self.PMxmin, self.PMxmax], [self.PMymin, self.PMymax]], \
										   norm = mplColors.LogNorm(), cmap = cm.Greys)
			ax2.contourf(x2D[:-1], y2D[:-1], pmG2D(xf, yf).T, cmap=cm.RdPu, bins = 20, \
						 norm=mplColors.LogNorm(), alpha = 0.3)

			ax1.set_xlim(self.PMxmin, self.PMxmax)
			ax2.set_xlim(self.PMxmin, self.PMxmax)
			ax2.set_ylim(self.PMymin, self.PMymax)
			ax3.set_ylim(self.PMymin, self.PMymax)
			#ax1.set_yscale("log")
			#ax1.set_ylim(1, 2*max(hx1D))
			#ax3.set_xscale("log")
			#ax3.set_xlim(1, 2*max(hy1D))
			#ax2.set_xlabel(r'$\mu_\alpha \cos(\delta)$ (mas yr$^{-1}$)', fontsize=16)
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
		Fc = models.Gaussian2D()
		Fc.parameters = pmG2D.parameters[0:6]
		self.PPM = Fc(x,y)/pmG2D(x,y)
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
								  np.nan_to_num(self.data['PPM'].filled(), nan=1)

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
		ax.plot(data[mask][x1] - data[mask][x2], data[mask][y],'.', color='deeppink')

		ax.set_ylim(22, 10)
		ax.set_xlim(-1, 3)
		ax.set_xlabel('(g_ps - i_ps)', fontsize=16)
		ax.set_ylabel('g_ps', fontsize=16)
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



	def runAll(self):
		self.getData()
		self.getRVMembers()
		self.getParallaxMembers()
		self.getPMMembers()
		self.combineMemberships()
		self.plotCMD()
		self.generatePhotFile()
		self.generateYamlFile()

		if (self.verbose > 0):
			print("done.")
