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

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos
from scipy.integrate import quadrature as quad
from scipy.stats import linregress

prop = mpl.font_manager.FontProperties(fname='/Library/Fonts/GillSans.ttc')

np.set_printoptions(linewidth=150, precision=3)

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
	bounds_eps = 1e-15														# we need some infinitesimal spacer to avoid evaluating the function at the discontinuity

	for j in range(N):

		k_low = bounds[j]+bounds_eps
		k_high = bounds[j+1]-bounds_eps

		try:
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

		kappa = np.linspace(0,n*k,10000)

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

		plt.plot(kappa*d/pi, trans(kappa),'r')
		plt.plot(kappa*d/pi, np.sign(trans(kappa)), 'k:')

		# for j in range(N):
		# 	plt.axvline(Ks[j]*d/pi)

		plt.xlabel(r'$\kappa d/\pi$')
		plt.axhline(0, c='k')
		plt.ylim(-5,5)
		plt.show()
		exit()

	return np.array(Bs)

def test_beta_marcuse():

	target_vals_TE = [0.25781,0.54916,1.21972,1.93825,2.66839,4.13075]
	target_vals_TM = [0.25207,0.51677,1.12809,1.84210,2.58934,4.08131]

	wl = 1

	n = 1.432
	k = 2*pi/wl

	kds = np.array([0.25,0.5,1.0,1.5,2.0,3.0])
	ds = kds/k

	print 'TE'
	for i,d in enumerate(ds):
		bTE = beta_marcuse(n,d,pol='TE',wl=wl)[0]
		print "%.5f, %.5f, %.5f" % (bTE * d, target_vals_TE[i], abs((bTE * d - target_vals_TE[i]) / target_vals_TE[i]))

	print
	print 'TM'

	for i,d in enumerate(ds):
		bTM = beta_marcuse(n,d,pol='TM',wl=wl)[0]
		print "%.5f, %.5f, %.5f" % (bTM * d, target_vals_TM[i], abs((bTM * d - target_vals_TM[i]) / target_vals_TM[i]))

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

def convergence_test_single():
	'''
	For a specific value of kd, sweep pmax and pres and plot the result
	'''

	kd = 0.15

	n = sqrt(20)

	res = np.arange(1,15,1) * 1e2
	p_max = np.arange(5,31,2.5)

	incident_mode = 0
	pol='TE'
	polarity = 'even'

	imax = 200

	plt.ion()

	fig, ax = plt.subplots(3,figsize=(10,8),sharex=True,sharey=True)
	plt.show()

	aos = np.zeros((len(res),len(p_max)),dtype=complex) * complex(np.nan,np.nan)
	accuracy = np.zeros((len(res),len(p_max)),dtype=complex) * complex(np.nan,np.nan)

	mag = ax[0].imshow(abs(aos), aspect='auto')
	divmag = make_axes_locatable(ax[0])
	cmag = divmag.append_axes("right", size="1%", pad=0.05)
	cbarmag = plt.colorbar(mag,cax=cmag,format='%.2f')

	phase = ax[1].imshow(np.angle(aos)/pi, aspect='auto')
	divphase = make_axes_locatable(ax[1])
	cphase = divphase.append_axes("right", size="1%", pad=0.05)
	cbarphase = plt.colorbar(phase,cax=cphase,format='%.2f')

	error = ax[2].imshow(abs(accuracy), aspect='auto')
	diverror = make_axes_locatable(ax[2])
	cerror = diverror.append_axes("right", size="1%", pad=0.05)
	cbarerror = plt.colorbar(error,cax=cerror,format='%.2f')

	ax[2].set_xlabel(r'Max$\{\rho\}$')
	ax[1].set_ylabel('res')


	for i,res_i in enumerate(res):
		for j,pmax_j in enumerate(p_max):

			a, acc = Reflection(kd,n,
													pol=pol,
													polarity=polarity,
													incident_mode=incident_mode,
													p_res=res_i,
													imax=imax,
													p_max=pmax_j,
													first_order=False,
													debug=False
													)

			try:
				aos[i,j] = a[0][0]
				accuracy[i,j] = acc
			except TypeError:
				pass

			# |ao|
			mag.set_data(abs(aos))
			mag.set_extent(getExtent(p_max,res))
			mag.autoscale()

			# phase ao
			phase.set_data(np.angle(aos)/pi)
			phase.set_extent(getExtent(p_max,res))
			phase.autoscale()

			# error
			error.set_data(abs(accuracy))
			error.set_extent(getExtent(p_max,res))
			error.autoscale()

			fig.canvas.draw()

	plt.ioff()
	plt.show()

def ReflectionWithHamidsCorrections(kd,n,incident_mode=0,pol='TE',polarity='even',
	p_max=20,p_res=1e3,imax=100,convergence_threshold=1e-5,first_order=False, debug=False, returnMode = None, returnItr = 0):

	'''
	Major code refactor for cleanliness. TE even modes only.
	'''

	debug = False

	m = incident_mode

	# constants
	wl = 10. # Set to 10 to match Gelin values for qt
	k = 2*pi/wl
	d = kd/k
	w = c*k
	eps = n**2

	# Solve slab for all modes
	N = numModes(n,1,kd)
	B = beta_marcuse(n,d,wl=wl,pol='TE',polarity='even',Nmodes=N,plot=False)

	# Define Wavevectors
	g  = sqrt(      -k**2 + B**2)						# transverse wavevectors in air for guided modes
	K  = sqrt(n**2 * k**2 - B**2)						# transverse wavevectors in high-index region for guided modes

	Bc = lambda p: sqrt(k**2 - p**2)
	o  = lambda p: sqrt((n*k)**2 - Bc(p)**2)

	# Mode Amplitude Coefficients
	P = 1

	A = sqrt(2*w*mu*P / (B*d + B/g)) # Tested for both even and odd

	Bt = lambda p: sqrt(2*w*mu*P / (pi*abs(Bc(p))))
	Br = lambda p: sqrt(2*p**2*w*mu*P / (pi*abs(Bc(p))*(p**2*cos(o(p)*d)**2 + o(p)**2*sin(o(p)*d)**2)))
	Dr = lambda p: 1/2. * exp(-1j*p*d) * (cos(o(p)*d) + 1j*o(p)/p * sin(o(p)*d))

	# Define Helper functions for integrals

	G = lambda m,p: 2 * k**2 * (eps-1) * A[m] * Bt(p) * cos(K[m]*d) * (g[m]*cos(p*d) - p*sin(p*d)) / ((K[m]**2 - p**2)*(g[m]**2 + p**2))

	def Ht(qr,a,b):

		'''
		qr is taken in as simply qr(p). To correspond to qr(p') in this 2D matrix formulation, it needs
		to be reshaped into a 2D matrix, whose values are dependent by row (0th axis), but not by column.
		'''
		qr = np.tile(qr,(len(qr),1)).transpose()

		return -qr * (B[m] - Bc(a)) * k**2 * (eps-1) * Bt(b) * Br(a) * (  sin( (o(a)+b) *d)/(o(a)+b)  +  sin( (o(a)-b) *d)/(o(a)-b)  )

	def Hr(qt,a,b):

		qt = np.tile(qt,(len(qr),1)).transpose()

		return -qt * (Bc(b) - Bc(a)) * k**2 * (eps-1) * Bt(b) * Br(a) * (  sin( (o(a)+b) *d)/(o(a)+b)  +  sin( (o(a)-b) *d)/(o(a)-b)  )


	# Define mesh of p values
	pmax = p_max*k
	pres = p_res
	p = np.linspace(1e-3,pmax,pres)



	# try using a higher resolution near the singularity
	# eps_k = 1e-2
	# sing_res = pres
	# p_l = np.linspace(1e-3,k-eps_k,pres/2.)
	# p_sing = np.linspace(k-eps_k,k+eps_k,sing_res)	
	# p_r = np.linspace(k+eps_k,pmax,pres/2.)

	# p = np.concatenate((p_l,p_sing,p_r))
	# p = np.unique(p) # remove duplicates


	# use Gauss-Chebyshev points for integral evaluation
	# weights, p = cheb.getPointsAndWeights(0,pmax,10)

	# plt.ioff()
	# plt.figure()
	# plt.plot(p)
	# plt.show()
	# exit()

	'''
	2D mesh of p values for performing integration with matrices. Rows correspond to
	the value of p', columns to p (for H(p',p)).

	Ex. H(p',p)[3,2] chooses the value of H with the 4th value of p' and the 3rd value of p.

	Integrating over p' would then amount to summing over axis=0. This sums the rows of the matrix H, producing 
	a row vector representing a function of p.
	'''
	p2,p1 = np.meshgrid(p,p) 

	# Test Helper Function evaluation on array objects
	# Works without vectorization for single kd input
	if debug:

		print 'Test axes of matrices - p1 should change with row (1st index):'
		print 'p1[0,0]:', p1[0,0]
		print 'p1[1,0]:', p1[1,0]
		print 'p1[0,1]:', p1[0,1]

		print '\nG'
		print G(0,p)
		print '\np1'
		print p1
		print '\np2'
		print p2
		print '\nH'
		print H(1,p1,p2)
		print '\nHpp'
		print H(1,p2,p2)

	# Define initial states for an, qr
	a = np.zeros(N, dtype=complex)
	qr = np.zeros(len(p), dtype=complex)

	'''
	Iteratively define qt,a,qr until the value of a converges to within some threshold.
	'''
	repeat = True
	converged = False
	delta_prev = delta_2prev = np.inf

	for i in range(imax):
		print '\nComputing iteration %u of %u' % (i+1,imax)

		if not repeat:
			break

		# TODO - process the lead terms before looping, since they are independent of the iteration
		
		integrand = (Ht(qr,p1,p2) - Ht(qr,p2,p2))/(p1**2 - p2**2) # blows up at p1=p2
		integrand = smoothMatrix(integrand)											# elminate singular points by using average of nearest neighbors.

		if debug:
			print integrand
			print np.trapz(integrand,dx=1, axis=0)
		
		# if i == 1:
		# 	plt.ioff()
		# 	plt.figure()

		# 	plt.plot(p/k,abs(integrand[0,:]),'r')
		# 	plt.plot(p/k,abs(integrand[:,0]),'b')

		# 	plt.figure()
		# 	# print sp.log10(abs(integrand[250,0]))
		# 	# print sp.log10(abs(integrand[250,1]))
		# 	# plt.imshow(sp.log10(abs(Ht(qr,p1,p2))), extent = getExtent(p/k,p/k))
		# 	plt.imshow(sp.log10(abs(integrand)), extent = getExtent(p/k,p/k))
		# 	plt.colorbar()
		# 	plt.show()
		# 	exit()

		qt = 1/(2*w*mu*P) * abs(Bc(p))/(B[m]+Bc(p)) * ( \
			2*B[m]*G(m,p) \
			+ np.sum([  (B[m]-B[j])*a[j]*G(j,p) for j in range(N) ], axis=0) \
			# + sp.integrate.quad(integrand, x=p, axis=0) \
			+ sp.integrate.simps(integrand, x=p, axis=0) \
			# + np.trapz(integrand, x=p, axis=0) \
			+ qr * (B[m]-Bc(p)) * pi * Bt(p)*Br(p)*2*np.real(Dr(p))
			)

		a_prev = a
		a = [1/(4*w*mu*P) * sp.integrate.simps(qt * (B[j]-Bc(p)) * G(j,p), x=p) for j in range(N)]; a = np.array(a);
		# a = [1/(4*w*mu*P) * np.trapz(qt * (B[j]-Bc(p)) * G(j,p), x=p) for j in range(N)]; a = np.array(a);

		integrand = (Hr(qt,p2,p1) - Hr(qt,p2,p2))/(p2**2 - p1**2) # blows up at p1=p2
		integrand = smoothMatrix(integrand)

		# if i == 0:
		# 	plt.ioff()
		# 	plt.figure()

		# 	plt.plot(p/k,abs(integrand[0,:]),'r')
		# 	plt.plot(p/k,abs(integrand[:,0]),'b')
		# 	plt.plot(p/k, Br(p),'g')

		# 	plt.figure()
		# 	# print sp.log10(abs(integrand[250,0]))
		# 	# print sp.log10(abs(integrand[250,1]))
		# 	plt.imshow(np.real((Hr(qt,p2,p1) - Hr(qt,p2,p2))/(p2 + p1)), extent = getExtent(p/k,p/k))
		# 	# plt.imshow(sp.log10(abs(Ht(qr,p1,p2))), extent = getExtent(p/k,p/k))
		# 	# plt.imshow(sp.log10(abs(integrand)), extent = getExtent(p/k,p/k))
		# 	plt.colorbar()
		# 	plt.show()
		# 	exit()

		qr = 1/(4*w*mu*P) * abs(Bc(p))/Bc(p) * sp.integrate.simps(integrand, x=p, axis=0)
		# qr = 1/(4*w*mu*P) * abs(Bc(p))/Bc(p) * np.trapz(integrand, x=p, axis=0)

		if returnMode and i == returnItr:
			# bail and return whatever value we're testing
			if returnMode == 'qt': return (p,qt)
			if returnMode == 'qr': return (p,qr)
			if returnMode == 'a_integral': return (p,qt * (B[j]-Bc(p)) * G(j,p))


		# Test for convergence
		delta = abs(a_prev-a)
		print 'Delta a:', np.amax(delta)
		if not np.any(delta > convergence_threshold):
		 repeat = False
		 converged = True			

		# if difference in am has been rising for 2 iterations, value is diverging. Bail.

		if np.amax(delta) > delta_prev and delta_prev > delta_2prev: break

		delta_2prev = delta_prev
		delta_prev = np.amax(delta)


	'''
	Loop completed. Perform Error tests
	'''

	shouldBeOne = 1/(4*w*mu*P) * np.trapz(qt*(B[m]+Bc(p))*G(m,p), x=p)

	'''
	Output results
	'''

	print "\n--- Outputs ---"
	if debug:
		print '|a|:', abs(a)
		print '<a:', np.angle(a)
		print 'shouldBeOne:', shouldBeOne

	if converged:
		return a, np.array(shouldBeOne)
	else:
		return a * np.nan, np.array(np.nan)

def TestHarness():

	# Define Key Simulation Parameters
	
	# To convert into the d = h/wl notation for dielectricSlab, use:
	# kd = d*pi/n

	kd = 0.8

	n = sqrt(20)

	res = [100,200,1000,2000]
	incident_mode = 0
	pol='TE'
	polarity = 'even'

	imax = 200
	p_max = 10

	method = ReflectionWithHamidsCorrections
	returnMode = "a_integral"
	returnItr = 1

	for i,r in enumerate(res):
		returnVal = method(kd,n,
										pol=pol,
										polarity=polarity,
										incident_mode=incident_mode,
										p_res=r,
										imax=imax,
										p_max=p_max,
										first_order=False,
										debug=False,
										returnMode = returnMode,
										returnItr = returnItr
										)

		plt.plot(returnVal[0],returnVal[1])

	plt.legend(res, loc='best')
	plt.show()

def main():

	# Define Key Simulation Parameters
	
	# kds = np.array([3.])
	
	# kds = np.array([0.209,0.418,0.628,0.837,1.04,1.25]) # TE Reference values
	# kds = np.array([0.314,0.418,0.628,0.837,1.04,1.25]) # TM Reference Values

	# To convert into the d = h/wl notation for dielectricSlab, use:
	# kd = d*pi/n

	# Note: If kd is too small, BrentQ will fail to converge.
	# kds = np.linspace(0.6,2.4,100)
	# kds = np.linspace(0.1,0.5,50)
	kds = np.linspace(1e-3,3,50)

	n = sqrt(20)

	res = 200
	incident_mode = 0
	pol='TE'
	polarity = 'even'

	imax = 50
	p_max = 10

	plt.ion()
	

	fig, ax = plt.subplots(3,figsize=(4.5,8))
	ax[2].axhline(1, color='k', ls = ':')
	plt.show()

	ams = []
	accuracy = []

	method = ReflectionWithHamidsCorrections

	for kdi,kd in enumerate(kds):

		print '\nkd:', kd

		a, acc = method(kd,n,
										pol=pol,
										polarity=polarity,
										incident_mode=incident_mode,
										p_res=res,
										imax=imax,
										p_max=p_max,
										first_order=False,
										debug=False
										)

		ams.append(a)
		accuracy.append(np.abs(acc))

		data = stackPoints(ams)
		data = np.array(data)

		[ax[0].plot(kds[:kdi+1], abs(data[i])        , color=colors[i], marker='o') for i,am in enumerate(data)]
		[ax[1].plot(kds[:kdi+1], np.angle(data[i])/pi, color=colors[i], marker='o') for i,am in enumerate(data)]
		ax[0].set_ylim(0,1.2)
		
		[ax[2].plot(kds[:kdi+1], accuracy, color='b', marker='o', lw=2) for i,am in enumerate(data)]
		ax[2].set_ylim(0,1.5)

		fig.canvas.draw()

	plt.ioff()
	plt.show()

	## Output for Comparison with Gelin's Table
	# print ams
	# print
	# print abs(ams)**2

	## export data
	output = np.vstack((kds,data,np.array(accuracy, dtype=complex)))
	print output

	sname = DATA_PATH + '/Rectangular Resonator/End Facet Reflection Coefs/endFacet_a' + str(incident_mode) + '_' + pol + '_' + polarity + '_' + str(n) + '.txt'
	np.savetxt(sname,output)

'''
TODO

plot qt*dp instead of qt when comparing to Gelin figure
bug when submitting multiple kds - returns nan
Fix Gelin error equation for odd modes
mode solver error for odd TE when second mode appears
use mag ao for convergence?
compare RCWA
output to file
optimization of resolution - pick one kd and sweep
fix error for int type kds
'''

if __name__ == '__main__':
  # test_beta_marcuse()
  # convergence_test_single()
  main()	
  # PrettyPlots()
	# TestHarness()