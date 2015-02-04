#!/usr/bin/env python

from __future__ import division

from EndFacet import *

import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt

import putil
import sys
import os

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos

pi = sp.pi
sqrt = sp.emath.sqrt

colors = ['r','g','b','#FFE21A','c','m']

# Set defaults for plt.imshow()
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

prop = mpl.font_manager.FontProperties(fname='/Library/Fonts/GillSans.ttc')

# Change reporting of numpy objects to lower float precision
np.set_printoptions(linewidth=150, precision=3)

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

def test_betaMarcuseAtKd():

	'''
	use for testing mode solver at a specific frequency kd
	'''

	# constants
	kd = 2.8
	n = sqrt(20)

	wl = 10. # Set to 10 to match Gelin values for qt
	k = 2*pi/wl
	d = kd/k
	
	# w = c*k
	# eps = n**2

	# Solve slab for all modes
	N = numModes(n,1,kd)
	print "Number of Modes from numModes: ", N
	B = beta_marcuse(n,d,wl=wl,pol='TE',polarity='even',Nmodes=N,plot=True)

def convergence_test_single():
	'''
	For a specific value of kd, sweep pmax and pres and plot the result
	'''

	kd = 2.

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

			a,dd,bb,acc = SolveForCoefficients(kd,n,
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
				aos[i,j] = a[0]
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

def test_slabSaveResults():

	n = sqrt(20)

	slab = Slab(n)
	slab.setMesh()
	slab.setIncidentMode(0)
	slab.polarization 	= 'TE'
	slab.polarity 			= 'even'

	for kd in np.linspace(1e-2,3,2):
		slab.setFrequencyFromKD(kd)
		slab.SolveForCoefficients()

	kds = slab.results['kds']
	a = slab.results['as']

	print kds
	print a

if __name__ == '__main__':
  # test_betaMarcuseAtKd()
  test_slabSaveResults()

