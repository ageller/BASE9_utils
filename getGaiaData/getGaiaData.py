# Python code to download Gaia data at a given location

from astroquery.gaia import Gaia
from astropy.modeling import models, fitting
from astropy.table import Column
import astropy.units as u

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mplColors
import matplotlib.cm as cm

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
		self.GaiaCatalog = "gaiaedr3.gaia_source" 
		self.PanSTARRSMatchCatalog = "gaiaedr3.panstarrs1_best_neighbour"
		self.PanSTARRSCatalog = "gaiadr2.panstarrs1_original_valid"
		self.TMASSMatchCatalog = "gaiaedr3.tmass_psc_xsc_best_neighbour"
		self.TMASSJoinCatalog = "gaiaedr3.tmass_psc_xsc_join"
		self.TMASSCatalog = "gaiadr1.tmass_original_valid"

		# maximum error that we will allow in a source to be retrieved (not sure what the best value is here)
		self.maxPMerror = 5 # mas/year

		# set to 1 or 2 to print out more (and even more) information
		self.verbose = 0
		
		# set to True to generate plots
		self.createPlots = True
		self.plotNameRoot = ''

		# columns to select
		self.columns = ['gaia.source_id',
			'gaia.ra',
			'gaia.dec',
			'gaia.pmra',
			'gaia.pmdec',
			'gaia.dr2_radial_velocity',
			'gaia.bp_rp',
			'gaia.phot_g_mean_mag',
			'gaia.parallax',
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

		# output
		self.SQLcmd = ''
		self.data = None # will be an astropy table

	def getData(self):
		columns = ', '.join(self.columns)

		if (self.verbose > 0):
			print("Retrieving Gaia data ... ")

		# for testing 
		# self.SQLcmd = f"SELECT TOP 5 {columns} " + \
		self.SQLcmd = f"SELECT {columns} " + \
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
			print(self.SQLcmd)
		job = Gaia.launch_job_async(self.SQLcmd, dump_to_file=False) #could save this to a file
		self.data = job.get_results()
		if (self.verbose > 2):
			print(self.data)


	def getRVMembers(self):
		# calculate radial-velocity memberships
		if (self.verbose > 0):
			print("Finding radial-velocity members ... ")
		
		x = self.data['dr2_radial_velocity']
		
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
			f.savefig(self.plotNameRoot + 'RVHist.pdf', format='PDF', bbox_inches='tight')
			
		#membership calculation
		Fc = models.Gaussian1D()
		Fc.parameters = rvG1D.parameters[0:3]
		self.PRV = Fc(x)/rvG1D(x)
		self.data['PRV'] = self.PRV

	def getParallaxMembers(self):
		# estimate memberships based on distance (could use Bailer Jones, but this simply uses inverted parallax)
		if (self.verbose > 0):
			print("Finding parallax members ... ")
			
		x = (self.data['parallax']).to(u.parsec, equivalencies=u.parallax()).to(u.parsec).value
		
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
			f.savefig(self.plotNameRoot + 'dHist.pdf', format='PDF', bbox_inches='tight')
			
		#membership calculation
		Fc = models.Gaussian1D()
		Fc.parameters = pa1D.parameters[0:3]
		self.PPa = Fc(x)/pa1D(x)
		self.data['PPa'] = self.PPa

	def getPMMembers(self):
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
			foo = models.Gaussian2D(*pmG2D.parameters[0:6])
			ax1.plot(x2D[:-1], np.sum(foo(xf, yf), axis=1), color='gray')
			foo = models.Gaussian2D(*pmG2D.parameters[6:6])
			ax1.plot(x2D[:-1], np.sum(foo(xf, yf), axis=1), color='darkslateblue', ls='dashed')
			ax1.axvline(pmG2D.parameters[1], color='tab:purple', ls='dotted')
			ax1.annotate(r'$\mu_\alpha$ =' + f'{pmG2D.parameters[1]:.1f}' + r'mas yr$^{-1}$', (pmG2D.parameters[1] + 0.05*(self.PMxmax - self.PMxmin), 0.95*max(hx1D)) )

			hy1D, y1D = np.histogram(y, bins=pmDecbins)
			ax3.step(hy1D, y1D[:-1], color='black')
			ax3.plot(np.sum(pmG2D(xf, yf), axis=0), y2D[:-1], color='deeppink', lw=5)
			foo = models.Gaussian2D(*pmG2D.parameters[0:6])
			ax3.plot(np.sum(foo(xf, yf), axis=0), y2D[:-1], color='gray')
			foo = models.Gaussian2D(*pmG2D.parameters[6:6])
			ax3.plot(np.sum(foo(xf, yf), axis=0), y2D[:-1], color='darkslateblue', ls='dashed')
			ax3.axhline(pmG2D.parameters[7], color='tab:purple', ls='dotted')
			ax3.annotate(r'$\mu_\delta$ =' + f'{pmG2D.parameters[7]:.1f}' + r'mas yr$^{-1}$', (0.95*max(hy1D), pmG2D.parameters[7] + 0.05*(self.PMymax - self.PMymin)), rotation=90)

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
			f.savefig(self.plotNameRoot + 'PMHist.pdf', format='PDF', bbox_inches='tight')

		#membership calculation
		Fc = models.Gaussian2D()
		Fc.parameters = pmG2D.parameters[0:6]
		self.PPM = Fc(x,y)/pmG2D(x,y)
		self.data['PPM'] = self.PPM

	def combineMemberships(self):
		# I'm not sure the best way to combine these
		# We probably want to multiple them together, but if there is no membership (e.g., in RV), then we still keep the star
		self.data['PRV'].fill_value = 1.
		#self.data['PPa'].fill_value = 1.  # it appears that this is not a masked column
		self.data['PPM'].fill_value = 1.

		self.data['membership'] = np.nan_to_num(self.data['PRV'].filled(), nan=1)*\
								  np.nan_to_num(self.data['PPa'], nan=1)*\
								  np.nan_to_num(self.data['PPM'].filled(), nan=1)

	def plotCMD(self):
		# I could specify the columns to use
		f, ax = plt.subplots(figsize=(5,8))
		ax.plot(self.data['g_mean_psf_mag'] - self.data['i_mean_psf_mag'], self.data['g_mean_psf_mag'],'.', color='lightgray')

		#members
		mask = (self.data['membership'] > self.membershipMin) 
		ax.plot(self.data[mask]['g_mean_psf_mag'] - self.data[mask]['i_mean_psf_mag'], self.data[mask]['g_mean_psf_mag'],'.', color='deeppink')

		ax.set_ylim(22, 10)
		ax.set_xlim(-1, 3)
		ax.set_xlabel('(g_ps - i_ps)', fontsize=16)
		ax.set_ylabel('g_ps', fontsize=16)
		f.savefig(self.plotNameRoot + 'CMD.pdf', format='PDF', bbox_inches='tight')

	def runAll(self):
		self.getData()
		self.getRVMembers()
		self.getParallaxMembers()
		self.getPMMembers()
		self.combineMemberships()
		self.plotCMD()