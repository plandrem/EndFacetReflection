#!/usr/bin/env python

from __future__ import division

import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt

import DielectricSlab as DS
from EndFacet import *

import putil
import sys
import os

from scipy import sin, cos, exp, tan, arctan, arcsin, arccos

pi = sp.pi
sqrt = sp.emath.sqrt

colors = ['r','b','g','#FFE21A','c','m']

# Set defaults for plt.imshow()
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

prop = mpl.font_manager.FontProperties(fname='/Library/Fonts/GillSans.ttc')

# Change reporting of numpy objects to lower float precision
np.set_printoptions(linewidth=150, precision=3)

def getFilename(incident_mode=0,polarization=None,polarity=None,n=None):
	return os.getcwd() + '/slabs/' + \
		'a' + str(incident_mode) + '_' + \
		polarization + '_' + \
		polarity + '_' + \
		str(n) + '.slab'

def generateSlabData():
  # Define Key Simulation Parameters
  
  # To convert into the d = h/wl notation for dielectricSlab, use:
  # kd = d*pi/n

  # Note: If kd is too small, BrentQ will fail to converge.
  kds = np.linspace(0.02,3,20)

  n = 4

  slab = Slab(n)
  slab.setIncidentMode(1)

  slab.polarization   = 'TM'
  slab.polarity       = 'even'

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

    slab.plotResults('a_mag',   ax=ax[0])
    slab.plotResults('a_angle', ax=ax[1])
    slab.plotResults('eq14',    ax=ax[2])

    fig.canvas.draw()

  plt.ioff()
  plt.show()

  sname = getFilename(n=slab.n,
  										polarity=slab.polarity,
  										polarization=slab.polarization,
  										incident_mode=slab.m*2)
  slab.save(sname)


def compareEndFacetWithRCWA():
	ds = np.linspace(0.02,3,100)
	ms  = np.array([0,2])
	
	pol = 'TM'
	
	fig,axs=plt.subplots(2,1,figsize=(6.5,5))
	
	# r,phi = effective_index_method(ds=ds,ms=m,pol=pol)
	
	for i,m in enumerate(ms):
		
		# load RCWA data
		rcwaRef = DS.Ref(  ds,pol=pol,mode=m,n=4)
		rcwaPhi = DS.phase(ds,pol=pol,mode=m,n=4) / 180.

		# load EndFacet data
		fname = getFilename(polarization=pol,
												n=4,
												incident_mode=m,
												polarity='even')
		slab = Slab.load(fname)

		kds = np.array(slab.results['kds'])
		slab_ds = kds * 4/pi

		a = slab.getCoefsForMode(m/2.)

		slabRef = abs(a)
		slabPhi = 1 - np.angle(a,deg=True) / 180.
		
		# Reflectance plots
		axs[0].plot(ds, rcwaRef, c=colors[i], lw=2)
		axs[0].plot(slab_ds, slabRef, c=colors[i], ls='', marker='o', markersize=6)

		# Phase plots
		axs[1].plot(ds, rcwaPhi, c=colors[i], lw=2)
		axs[1].plot(slab_ds, slabPhi, c=colors[i], ls='', marker='o', markersize=6)
			
		# Titles
		axs[0].set_ylabel('|R|',fontsize=16)
		# axs[0].set_ylim(0,1)
		
		axs[1].set_ylabel(r'$\phi/\pi$',fontsize=16)
		axs[1].set_xlabel(r'$d/\lambda$',fontsize=16)
		# axs[1].legend(('Effective Index','RCWA'))
		# axs[1].set_ylim(0,0.5)
		
	plt.show()
	

if __name__ == '__main__':
	# generateSlabData()
	compareEndFacetWithRCWA()
