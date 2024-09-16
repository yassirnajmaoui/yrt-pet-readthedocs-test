#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module used to extract the 2D Radon parameters of the radial basis of the Savant DOI 
scanner in its default configuration.
"""


#########################################################################################
# Imports
import numpy as np
from DataFlow.histogramBuilder import HistogramBuilder


########################################################################################
# List of methods created to build the features.

def createDetectorPairOfRadialBasis():
	"""
	Def.: Create the projection detector Id of the radial basis, seperated in two cases.
	      First, sameAxial refer to projections with detector pair sharing the same 
		  axial Id. Second, diffAxial, refer to projections with detector pair that have
		  different axial Id.
	Return:
		projDetIdRadialBasis_sameAxial (2D numpy array, Integer)[-1, 2]
		projDetIdRadialBasis_diffAxial (2D numpy array, Integer)[-1, 2]
	"""
	# Generate the two basic case: same and different ring.
	histoBuilder = HistogramBuilder()
	histoBuilder.setHistogramConfigToSavantDefault()
	histoBuilder.diffAxialId = 1
	radialBasisProjDetId_sameRing, radialBasisProjDetId_diffRing = \
	                                                histoBuilder.genRadProjDetIdBasis()
	
	nbDetPerLayer = histoBuilder.nbDetRadial * histoBuilder.nbDetAxial	
	nbDetPerRadial = histoBuilder.nbDetRadial
	
	# The method that create that part is too much imbuded in histogram builder so we 
	# make it here by hand.
	addLayerIdToSecondDetector = np.array([0, nbDetPerLayer])[np.newaxis, :]
	addAxialIdToSecondDetector = np.array([0, nbDetPerRadial])[np.newaxis, :]
	# Stack layer [0, 0], [0, 1] and [1, 1]
	projDetIdRadialBasis_sameAxial = np.vstack((radialBasisProjDetId_sameRing, \
	                       radialBasisProjDetId_diffRing + addLayerIdToSecondDetector, \
	                       radialBasisProjDetId_sameRing + nbDetPerLayer))
	
	# Stack layer [0, 0], [0, 1], [1, 0] and [1, 1]
	projDetIdRadialBasis_diffAxial = np.vstack((\
	                       radialBasisProjDetId_diffRing + addAxialIdToSecondDetector, \
	                       radialBasisProjDetId_diffRing + addLayerIdToSecondDetector \
	                            + addAxialIdToSecondDetector, \
	                       radialBasisProjDetId_diffRing[:, ::-1] \
	                            + addLayerIdToSecondDetector \
	                            + np.array([nbDetPerRadial, 0])[np.newaxis, :], \
	                       radialBasisProjDetId_diffRing + nbDetPerLayer \
	                            + addAxialIdToSecondDetector))	
						 
	return projDetIdRadialBasis_sameAxial, projDetIdRadialBasis_diffAxial


def genDetPosOfProj(_projDetId, _detLutFile):
	"""
	Def.: Generate the detector position of the projections specified.
	@_projDetId (2D numpy array, Integer): Detector Id of the projections.
	@_detLutFile (String): Path to the file holding the center of the detectors.
	Return: (3D numpy array, Float)[-1, 2, 3]
	"""
	detPos = np.fromfile(_detLutFile, dtype='float32').reshape((-1, 6))[:, :3]

	projDetPos = np.zeros((_projDetId.shape[0], 2, 3))
	projDetPos[:, 0, :] =  detPos[_projDetId[:, 0], :]
	projDetPos[:, 1, :] =  detPos[_projDetId[:, 1], :]
	
	return projDetPos


def computeProjRadonParamsFromDetectorPos(_projDetPos):
	"""
	Def.: Compute the 2D Radon parameters of the projection specified. 
	@_projDetPos (3D numpy array, Float)[-1, 2, 3]: The position of the detector pair of
		each projection.
	Return:
		beta (1D numpy array, Float)
		distToCenter (1D numpy array, Float)
	"""
	proj2DVec = _projDetPos[:, 0, :2] - _projDetPos[:, 1, :2] 
	proj2DVecNorm = proj2DVec / np.linalg.norm(proj2DVec, axis=1)[:, np.newaxis]
	# When the projection vector orientaion goes in negative in the y-axis, we need to 
	# invert the angle.
	proj2DVecNorm[np.where(proj2DVecNorm[:, 1] < 0.0)] *= -1.0
	
	beta = np.arccos(proj2DVecNorm[:, 0]) / np.pi * 180
	
	distToCenter = np.diff(_projDetPos[:, 0, 1::-1] \
							* _projDetPos[:, 1, :2], axis=1)[:, 0] \
					/ np.linalg.norm(proj2DVec, axis=1)	
	# Distance relation is inverted when angle is greater than 90 degree.
	distToCenter[np.where(beta >=90)] *= -1.0
							
	return beta, distToCenter


def saveRadonParams(_beta, _u, _fileName):
	"""
	Def.: Save the 2D radon parameters to a file.
	@_beta (1D numpy array, Float): Array of the projection angle, in degree.
	@_u (1D numpy array, Float): Array of the projection position, in mm.
	@_fileName (String): Path and file name where to save.
	"""
	radonParam = np.hstack((_beta[:, np.newaxis], _u[:, np.newaxis]))
	radonParam.astype('float32').tofile(_fileName)
	
	
def loadRadonParams(_fileName):
	"""
	Def.: Load the 2D radon parameters from a file.
	@_fileName (String): Path and file name to load the data from.
	Return: (2D numpy array, Float)[-1, 2]
	"""
	radonParam = np.fromfile(_fileName, dtype='float32').reshape((-1, 2))
	
	return radonParam

	
def genFinalResults():
	"""
	Def.: Create and save the 2D radon parameters for the radial basis of the default 
	      savant DOI configuration.  
	"""
	
	detLutFile = "scannerConfig/savantDOI.glut"
	resultsFileName_sameAxial = "savantDoi_radonParam_radialBasis_sameAxial.dat"
	resultsFileName_diffAxial = "savantDoi_radonParam_radialBasis_diffAxial.dat"
	
	projDetIdRadialBasis_sameAxial, projDetIdRadialBasis_diffAxial = \
	                                        createDetectorPairOfRadialBasis()
	
	projDetPosRadialBasis_sameAxial = genDetPosOfProj(projDetIdRadialBasis_sameAxial, \
	                                                  detLutFile)
	projDetPosRadialBasis_diffAxial = genDetPosOfProj(projDetIdRadialBasis_diffAxial, \
	                                                  detLutFile)
	
	beta_radialBasis_sameAxial, u_radialBasis_sameAxial =\
	              computeProjRadonParamsFromDetectorPos(projDetPosRadialBasis_sameAxial)
	
	beta_radialBasis_diffAxial, u_radialBasis_diffAxial =\
                  computeProjRadonParamsFromDetectorPos(projDetPosRadialBasis_diffAxial)
	
	saveRadonParams(beta_radialBasis_sameAxial, u_radialBasis_sameAxial, \
					resultsFileName_sameAxial)
	saveRadonParams(beta_radialBasis_diffAxial, u_radialBasis_diffAxial, \
					resultsFileName_diffAxial)
	
	
#########################################################################################
# List of methods used to use this module as a script.
# None.


#########################################################################################
# Main : We use this to make the main usable has a script and a method.
#########################################################################################

if __name__== "__main__":
	"""
	Currently, this module is most likely a one time use so their is no need to make it
	work as a script.
	"""
	print("The main does nothing!")
	# genFinalResults()
