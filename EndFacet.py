# !/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import os
import time
import dill as pickle

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos
from scipy.integrate import quadrature as quad
from scipy.stats import linregress

prop = mpl.font_manager.FontProperties(fname='/Library/Fonts/GillSans.ttc')

np.set_printoptions(linewidth=150, precision=8)

pi = sp.pi
sqrt = sp.emath.sqrt

mu = 1.
eo = 1.
c  = 1.

# units
cm = 1e-2
um = 1e-6
nm = 1e-9

colors = ['r','g','b','#FFE21A','c','m']

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

PATH = '/Users/Patrick/Documents/PhD/'
DATA_PATH = PATH + '/DATA/'

def cot(x):
	return 1/tan(x)

def getExtent(xs,ys):
	x0 = np.amin(xs)
	y0 = np.amin(ys)
	x1 = np.amax(xs)
	y1 = np.amax(ys)
	return [x0,x1,y0,y1]
	
def getColorSeries(cmap,n):

	'''
	returns list of n rgba color values for plotting data with matplotlib.
	'''
	norm = mpl.colors.Normalize(vmin=0, vmax=n)
	sm = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)

	return [sm.to_rgba(i) for i in range(n)]

def stackPoints(arrays):
	# Find largest length
	N = 0
	for a in arrays:
		if len(a) > N: N = len(a)

	result = np.zeros((N,len(arrays)), dtype=complex)
	# Pad arrays and append to output
	for i,a in enumerate(arrays):
		pad = np.ones(N - len(a)) * complex(np.nan,np.nan)
		result[:,i] = np.append(a,pad)
	return result

def smoothMatrix(M):
	'''
	Helper function for Reflection. Takes integrand matrix, M, and for each point on the main
	diagonal sets the average of the two adjacent indices on axis 0.

	M must be square
	'''
	N = len(M[:,0])

	for n in range(N):
		if n==0: M[n,n] = M[n+1,n]
		elif n > 0 and n < N-1: M[n,n] = (M[n+1,n] + M[n-1,n]) / 2.
		elif n==N-1: M[n,n] = M[n-1,n]
	
	return M

def numModes(ncore,nclad,kd, polarity='even'):
	'''
	Returns number of allowed modes for a slab waveguide of thickness d
	See Marcuse, Light Transmission Optics, p. 310 for graphical origin
	'''
	
	C = 0 if polarity=='even' else pi/2.
	return np.ceil((sqrt(ncore**2 - nclad**2)*kd - C) / pi).astype(int)

def beta_marcuse(n,d,wl=1.,pol='TM',polarity='even',Nmodes=None,plot=False):

	'''
	Based on theory in Marcuse, 1970 "Radiation Losses in Tapered Dielectric Waveguides"

	Computes beta for dielectric slab in air (can be modified for asymmetric slab)

	Returns array (length Nmodes) of wavevectors - if fewer than Nmodes exist, array is padded 
	with np.nan

	0th index result corresponds to fundamental

	Currently only valid for even modes!

	INPUTS:
	n - index of slab
	d - half-thickness of slab

	'''
	# # Use to force plot/exit when debugging
	# plot=True

	even = (polarity=='even')

	k = 2*pi/wl
	gamma = lambda x: sqrt((n*k)**2 - x**2 - k**2)
	

	# Define the transcendental function for the allowed transverse wavevectors. Note that this is a complex function
	# but we only need to look for zeros on the real axis as long as our refractive index is purely real.
	C = n**2 if pol == 'TM' else 1.

	if even:
		trans = lambda K: np.real( tan(K * d) - C   * (gamma(K)/K) )
	else:
		trans = lambda K: np.real( tan(K * d) + 1/C * (K/gamma(K)) )


	# Find zero crossings, then use the brentq method to find the precise zero crossing 
	# between the two nearest points in the array

	Ks = np.array([])

	# set markers for the k values where tan(kd) becomes discontinuous:
	# ie. kd = (n + 1/2)*pi
	N = numModes(n,1,k*d,polarity=polarity)				# this is the maximum number of modes the system supports
	bounds = [(_n+0.5)*pi/d for _n in range(N)]
	bounds.insert(0,0)
	bounds_eps = 1e-9														# we need some infinitesimal spacer to avoid evaluating the function at the discontinuity

	for j in range(N):

		k_low = bounds[j]+bounds_eps
		k_high = bounds[j+1]-bounds_eps

		try:
			# I have tested this root-finding methods at other tolerances and against other scipy options; there was 
			# no difference I could see within 8 decimal places (I didn't test finer resolution).
			k_zeroCrossing = sp.optimize.brentq(trans,k_low,k_high)

		except ValueError:
			print "BrentQ error:"
			print k_low, k_high
			print trans(k_low), trans(k_high)
			k_zeroCrossing = np.nan
			plot=True

		Ks = np.append(Ks,k_zeroCrossing)

	# Done finding zero crossings for transcendental function.




	# Convert from transverse wavevector to propagation constant, Beta
	Bs = sqrt((n*k)**2 - Ks**2)

	# Truncate or pad output as necessary
	if len(Bs) < Nmodes:
		pad = np.zeros(Nmodes - len(Bs)) * np.nan
		Bs = np.hstack((Bs,pad))
		
	elif len(Bs) > Nmodes:
		Bs = Bs[:Nmodes] 

	# Plots for debugging
	if plot:

		plt.ioff()

		kappa = np.linspace(0,2*bounds[-1]-bounds_eps,10000)
		# kappa = np.linspace(0,bounds[-1]-bounds_eps,10000)

		# print 'Number of modes:', Nmodes
		print Ks*d/pi
		# print 'Zero Crossings:', kappa[np.nonzero(diff)[0]] * d/pi

		plt.figure()

		plt.plot(kappa*d/pi, tan(kappa*d),'b')

		if even:
			plt.plot(kappa*d/pi, C * sqrt(n**2*k**2 - kappa**2 - k**2)/kappa,'g')
		else:
			plt.plot(kappa*d/pi, 1/C * (-kappa)/gamma(kappa).real,'g')
			# plt.plot(kappa*d/pi, 1/C * (-kappa)/sqrt(n**2*k**2 - kappa**2 - k**2))

		plt.plot(kappa*d/pi, trans(kappa).real,'r')
		plt.plot(kappa*d/pi, trans(kappa).imag,'r:')

		# for j in range(N):
		# 	plt.axvline(Ks[j]*d/pi)

		plt.xlabel(r'$\kappa d/\pi$')
		plt.axhline(0, c='k')
		plt.ylim(-5,5)
		plt.show()
		exit()

	return np.array(Bs)

class Slab():

	def __init__(self,n,d=1):

		# Structural Properties
		self.n = n
		self.eps = n**2
		self.d = d

		# Optical Properties
		self.polarity = 'even'
		self.polarization = 'TE'
		self.m = 0
		self.k = None
		self.wl = None
		self.kd = None
		self.P = 1
		self.w = 1

		self.converged = False

		# Data Storage and Retrieval
		self.results = {
			'n' : n,
			'polarity': self.polarity,
			'polarization': self.polarization,
			'incident_mode': self.m,

			'pmax': None,
			'pmin': None,
			'pres': None,

			'kds': [],
			'ps' : [],
			'as' : [],
			'ds' : [],
			'bs' : [],
			'eq14': [],
			'Zs':[]
		}

	def setMesh(self,pmin=1e-9,pmax=15,pres=400):
		'''
		pmax is given as a multiple of k.
		'''
		self.pmin = pmin
		self.pmax = pmax
		self.pres = pres

		self.results['pmin'] = pmin
		self.results['pmax'] = pmax
		self.results['pres'] = pres

		if self.k:
			p = self.p = np.linspace(pmin, pmax*self.k, pres)
			self.p2,self.p1 = np.meshgrid(p,p)

	def setFrequencyFromKD(self,kd):
		self.kd = kd
		self.wl = 10
		self.k = 2*pi/self.wl
		self.d = kd/self.k
		self.w = c*self.k

		# update p values
		p = self.p = np.linspace(self.pmin, self.pmax * self.k, self.pres)
		self.p2,self.p1 = np.meshgrid(p,p)

		self.reset()

	def reset(self):
		'''
		sets output values to nan in case user trys querying them before they have
		been solved.
		'''
		self.a = [np.nan]
		self.dd = [np.nan]
		self.bb = [np.nan]

	def setIncidentMode(self,m):
		self.m = m

	def solveForModes(self):
		print "\nRunning Mode Solver..."
		N = self.N = numModes(self.n, 1, self.kd) # assumes external index is 1
		self.B = beta_marcuse(self.n,
										 self.d,
										 wl=self.wl,
										 pol=self.polarization,
										 polarity=self.polarity,
										 Nmodes=N,
										 plot=False)


		print "N:", N
		print "Betas:", self.B


	def setWavevectors(self):
		'''
		Should be called after setting frequency
		'''
		n = self.n
		k = self.k
		B = self.B

		self.g  = sqrt(      -k**2 + B**2)						# transverse wavevectors in air for guided modes
		self.K  = sqrt(n**2 * k**2 - B**2)						# transverse wavevectors in high-index region for guided modes

	def setArrays(self):
		'''
		Should be called after setting frequency and mesh
		'''
		p = self.p
		p2 = self.p2

		self.Zp = self.Z(p)
		self.Zp2 = self.Z(p2)

	def Bc(self,p):
		n = self.n
		k = self.k
		return sqrt(k**2 - p**2) * (2 * (sqrt(k**2 - p**2).real > 0) - 1) # Second term is Hamid's correction - not sure why this is necessary yet

	def o(self,p):
		n = self.n
		k = self.k
		Bc = self.Bc
		return sqrt((n*k)**2 - Bc(p)**2)

	'''
	Mode Amplitude Coefficient Functions
	'''

	def A(self):
		B = self.B
		g = self.g
		w = self.w
		d = self.d
		P = self.P

		return sqrt(2*w*mu*P / (B*d + B/g))

	def Bt(self,p):
		Bc = self.Bc
		w = self.w
		P = self.P

		return sqrt(2*w*mu*P / (pi*abs(Bc(p))))

	def Br(self,p):
		Bc = self.Bc
		o = self.o
		w = self.w
		d = self.d
		P = self.P

		return sqrt(2*p**2*w*mu*P / (pi*abs(Bc(p))*(p**2*cos(o(p)*d)**2 + o(p)**2*sin(o(p)*d)**2)))

	def Dr(self,p):
		o = self.o
		d = self.d

		return 1/2. * exp(-1j*p*d) * (cos(o(p)*d) + 1j*o(p)/p * sin(o(p)*d))

	def Z(self,p):
		w = self.w
		Bc = self.Bc

		alpha = 1 * (p <= self.k) + 1j * (p > self.k)
		return alpha * w * mu / abs(Bc(p))

	'''
	Mode Amplitude Coefficient Functions
	'''

	def G(self,m,p):
		k = self.k
		g = self.g
		K = self.K
		A = self.A()
		Bt = self.Bt
		d = self.d
		eps = self.eps

		return 2 * k**2 * (eps-1) * A[m] * Bt(p) * cos(K[m]*d) * (g[m]*cos(p*d) - p*sin(p*d)) / ((K[m]**2 - p**2)*(g[m]**2 + p**2))

	def Ht(self,qr,p1,p2):
		k = self.k
		o = self.o
		B = self.B
		Bc = self.Bc
		Br = self.Br
		Bt = self.Bt
		d = self.d
		eps = self.eps
		m = self.m

		'''
		qr is taken in as simply qr(p). To correspond to qr(p') in this 2D matrix formulation, it needs
		to be reshaped into a 2D matrix, whose values are dependent by row (0th axis), but not by column.
		'''
		qr = np.tile(qr,(len(qr),1)).transpose()

		return -qr * (B[m] - Bc(p1)) * k**2 * (eps-1) * Bt(p2) * Br(p1) * (  sin( (o(p1)+p2) *d)/(o(p1)+p2)  +  sin( (o(p1)-p2) *d)/(o(p1)-p2)  )

	def Hr(self,qt,p1,p2):
		k = self.k
		o = self.o
		B = self.B
		Bc = self.Bc
		Br = self.Br
		Bt = self.Bt
		d = self.d
		eps = self.eps
		m = self.m

		qt = np.tile(qt,(len(qt),1)).transpose()

		return qt * (Bc(p2) - Bc(p1)) * k**2 * (eps-1) * Bt(p2).conj() * Br(p1).conj() * (  sin( (o(p1)+p2) *d)/(o(p1)+p2)  +  sin( (o(p1)-p2) *d)/(o(p1)-p2)  )

	def update_bb(self):
		w = self.w
		P = self.P
		dd = self.dd
		Zp = self.Zp
		Zp2 = self.Zp2
		B = self.B
		Bc = self.Bc
		Bt = self.Bt
		Br = self.Br
		Dr = self.Dr
		G = self.G
		p = self.p
		p1 = self.p1
		p2 = self.p2
		a = self.a
		m = self.m
		N = self.N

		integrand = self.Ht(dd*Zp,p1,p2)/(p1**2 - p2**2) 	# blows up at p1=p2
		integrand = smoothMatrix(integrand)				# elminate singular points by using average of nearest neighbors.

		self.bb = 1/(2*w*mu*P) * abs(Bc(p))/(B[m]+Bc(p)) * ( \
			2*B[m]*G(m,p) /Zp \
			+ np.sum([  (B[m]-B[j])*a[j]*G(j,p) for j in range(N) ], axis=0)/Zp \
			+ np.trapz(integrand/Zp2, x=p, axis=0) \
			+ dd * (B[m]-Bc(p)) * pi * Bt(p)*Br(p)*2*np.real(Dr(p))
			)

	def update_a(self):
		w = self.w
		P = self.P
		bb = self.bb
		Zp = self.Zp
		Zp2 = self.Zp2
		B = self.B
		Bc = self.Bc
		G = self.G
		p = self.p
		N = self.N

		self.a_prev = self.a
		self.a = np.array(
		 [1/(4*w*mu*P) * np.trapz(bb*Zp * (B[j]-Bc(p)) * G(j,p).conj(), x=p) for j in range(N)]
		 )


	def update_dd(self):
		w = self.w
		P = self.P
		bb = self.bb
		Zp = self.Zp
		Zp2 = self.Zp2
		Bc = self.Bc
		p = self.p
		p1 = self.p1
		p2 = self.p2

		integrand = self.Hr(bb*Zp,p2,p1)/(p2**2 - p1**2) # blows up at p1=p2
		integrand = smoothMatrix(integrand)

		self.dd = 1/(4*w*mu*P) * abs(Bc(p))/Bc(p) * np.trapz(integrand/Zp2, x=p, axis=0)

	def equation14errorTest(self):
		w = self.w
		P = self.P
		bb = self.bb
		Zp = self.Zp
		B = self.B
		Bc = self.Bc
		G = self.G
		p = self.p
		m = self.m

		if not self.converged: return np.nan

		return 1/(4*w*mu*P) * np.trapz(bb*Zp*(B[m]+Bc(p))*G(m,p), x=p)


	def SolveForCoefficients(self, imax=200, initial_a=None, initial_d=None):

		'''
		This method is the heart of Gelin's algorithm, in which we iteratively define
		the values of the scattering amplitudes. Note that the names bb and dd are used 
		in place of b(p) and d(p), to avoid naming conflicts between dd and d, the height
		of the slab.

		Currently, bb and dd are switched from their definitions in Gelin's paper.
		'''

		'''
		Must have set frequency at this point. Prepare all remaining slab
		properties for simulation
		'''

		self.solveForModes()
		self.setWavevectors()		
		self.setArrays()

		N = self.N
		p = self.p
		m = self.m
		kd = self.kd

		'''
		Make sure the current source mode is supported at this frequency
		'''
		if m >= N:
			print "Mode %u not supported by slab at frequency kd = %.3f" % (m, kd)
			return False

		'''
		Define initial states for an, dd
		'''

		if initial_a != None:
			if initial_a.any() == np.nan:
				# previous iteration failed; start from scratch
				a = np.zeros(N, dtype=complex)
			else:
				a = initial_a
				if len(a) < N:
					# pad with zeros
					for _n in range(N-len(a)):
						a = np.append(a,0)
		else:
			a = np.zeros(N, dtype=complex)

		if initial_d != None:
			if initial_a.any() == np.nan:
				# previous iteration failed; start from scratch
				dd = np.zeros(len(p), dtype=complex)
			else:
				dd = initial_d
		else:
			dd = np.zeros(len(p), dtype=complex)

		# store arrays to parent slab class 
		self.a = a
		self.bb = None
		self.dd = dd

		'''
		Iteratively define qt,a,qr until the value of a converges to within some threshold.
		Returns boolean indicating if algorithm converged.
		'''

		converged = False
		delta_prev = delta_2prev = np.inf

		for i in range(imax):
			print '\nComputing iteration %u of %u' % (i+1,imax)

			# TODO - process the lead terms before looping, since they are independent of the iteration
			
			self.update_bb()
			self.update_a()
			self.update_dd()

			# Test for convergence
			delta = abs(self.a_prev-self.a)
			print 'Delta a:', delta
			if not np.any(delta > 1e-5):
			 converged = True
			 break

			# if difference in a has been rising for 2 iterations, value is diverging. Bail.

			if np.amax(delta) > delta_prev and delta_prev > delta_2prev: break

			# record difference for this iteration
			delta_2prev = delta_prev
			delta_prev = np.amax(delta)

		self.converged = converged

		if converged: self.storeResults()

		return converged

	def storeResults(self):
		res = self.results

		res['kds'].append(self.kd)
		res['ps'].append(self.p)
		res['as'].append(self.a)
		res['bs'].append(self.bb)
		res['ds'].append(self.dd)
		res['eq14'].append(self.equation14errorTest())

		# useful for calculating things later
		res['Zs'].append(self.Z)

		self.results = res

	def plotResults(self, mode, ax=None, show=False):
		'''
		mode options:
		'a_angle': complex angle of a coefficients
		'a_mag'  : magnitude of a coefficients
		'b'			 : real + imaginary parts of b coefficients
		'd'			 : real + imaginary parts of d coefficients
		'eq14'	 : Gelin equation 14 error test value
		'''

		# if no axes given, create a new figure for plotting
		if not ax:
			fig, ax = plt.subplots(1,figsize=(7,5))

		kds = self.results['kds']

		if mode in ['a_angle','a_mag']:			
			data = stackPoints(self.results['as'])
			data = np.array(data)

			if mode == 'a_angle':
				data = np.angle(data)/pi
			else:
				data = np.abs(data)

			colors = getColorSeries('jet',data.shape[0])

			[ax.plot(kds, data[i], color=colors[i], marker='o') for i,am in enumerate(data)]

		elif (mode == 'b' or mode == 'd'):
			# get generalized coefficient results
			cs = self.results['bs'] if mode == 'b' else self.results['ds']
			ps = self.results['ps']
			Zs = self.results['Zs']

			colors = getColorSeries('jet',len(kds))

			for _i,kd in enumerate(kds):
				Z0 = Zs[_i](0)

				ax.plot(ps[_i],np.real(cs[_i]*Z0),c=colors[_i],ls='-')
				ax.plot(ps[_i],np.imag(cs[_i]*Z0),c=colors[_i],ls=':')

		elif mode == 'eq14':			
			data = np.array(self.results['eq14'])
			ax.plot(kds, abs(data), color='b', marker='o')
			ax.axhline(1,c='k',ls=':')

		else:
			print "ERROR: Unrecognized plotting mode received!"


		if show:
			plt.show()

		return ax

	def save(self,filename="slab_data.slab"):
		with open(filename, 'wb') as f:
			pickle.dump(self,f)

	@staticmethod
	def load(filename="slab_data.slab"):
		with open(filename, 'rb') as f:
			return pickle.load(f)

		# res = self.results = unpickler.load(open(filename, 'rb'))

		# self.n = res['n']
		# self.polarity = res['polarity']
		# self.polarization = res['polarization']
		# self.m = res['incident_mode']
		# self.setMesh(res['pmin'],res['pmax'],res['pres'])


def PrettyPlots():

	n = sqrt(20)
	
	incident_mode = 0
	pol='TE'
	polarity = 'even'


	fname = DATA_PATH + '/Rectangular Resonator/End Facet Reflection Coefs/endFacet_a' + str(incident_mode) + '_' + pol + '_' + polarity + '_' + str(n) + '.txt'

	with open(fname,'r') as f:
		s = f.read()

	s = s.replace('+-','-')
	s = s.replace('(','')
	s = s.replace(')','')

	with open(DATA_PATH + '/endfacet_temp.txt','w') as f:	
		f.write(s)


	data = np.loadtxt(DATA_PATH + '/endfacet_temp.txt', dtype=complex)
	
	N = data.shape[0] - 2

	kd = data[0,:]
	err = data[-1,:]

	idx = np.arange(N) + 1

	fig, ax = plt.subplots(2,sharex=True,figsize=(4.5,7))

	for i in idx:
		ax[0].plot(kd,  abs(data[i,:])        , color=colors[i-1], lw=2)
		ax[1].plot(kd, -np.angle(data[i,:])/pi, color=colors[i-1], lw=2)

	ax[1].axhline(0, color='k', ls=':')

	# ax[0].set_xlim(0.6,1.5)
	ax[0].set_ylim(0,1.1)
	
	# ax[1].set_ylim(1/6.,2/3.)

	ax[1].set_xlabel(r'$k_{o}d$')

	ax[0].set_ylabel(r'$|a_{n}|$', fontsize=14)
	ax[1].set_ylabel(r'$\angle a_{n}$', fontsize=14)

	labels = [r'$a_{%u}$' % (2*j) for j in range(3)]
	ax[1].legend(labels, loc='lower left')

	ax[0].set_title(pol + ' - ' + r'Source Mode: $a_{%u}$' % 2*incident_mode, fontproperties=prop, fontsize=18)

	plt.tight_layout()
	plt.show()

def main():

	# Define Key Simulation Parameters
	
	# kds = np.array([2.7,2.8])
	# kds = np.array([0.628])
	
	# kds = np.array([0.209,0.418,0.628,0.837,1.04,1.25]) # TE Reference values
	# kds = np.array([0.314,0.418,0.628,0.837,1.04,1.25]) # TM Reference Values

	# To convert into the d = h/wl notation for dielectricSlab, use:
	# kd = d*pi/n

	# Note: If kd is too small, BrentQ will fail to converge.
	kds = np.linspace(1e-2,3,50)
	# kds = np.linspace(2.8,2.9,100)

	n = sqrt(20)

	slab = Slab(n)
	slab.setIncidentMode(1)

	slab.polarization 	= 'TE'
	slab.polarity 			= 'even'

	slab.setMesh()
	
	plt.ion()
	fig, ax = plt.subplots(3,figsize=(4.5,8))
	plt.show()

	ams = []
	accuracy = []

	for kdi,kd in enumerate(kds):

		print '\nkd:', kd
		slab.setFrequencyFromKD(kd)
		slab.SolveForCoefficients()

		slab.plotResults('a_mag',	  ax=ax[0])
		slab.plotResults('a_angle', ax=ax[1])
		slab.plotResults('eq14',    ax=ax[2])

		fig.canvas.draw()

	plt.ioff()
	plt.show()
	output = np.vstack((kds,data,np.array(accuracy, dtype=complex)))

	sname = DATA_PATH + '/Rectangular Resonator/End Facet Reflection Coefs/endFacet_a' + str(incident_mode) + '_' + pol + '_' + polarity + '_' + str(n) + '.txt'
	np.savetxt(sname,output)

'''
TODO

plot qt*dp instead of qt when comparing to Gelin figure
bug when submitting multiple kds - returns nan
Fix Gelin error equation for odd modes
compare RCWA
output to file
fix error for int type kds
replace A() with set_A() and store result to self.A
'''

if __name__ == '__main__':
  ### MAIN FUNCTIONS ###
  main()	
	# PrettyPlots()
