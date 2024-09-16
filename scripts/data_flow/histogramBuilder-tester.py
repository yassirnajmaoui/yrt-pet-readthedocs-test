#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module used to test the histogramBuilder module.

TODO:
	- Write more comments!
	- Clean the old testing function.
	- Make the script work with options at the place of modifying each time.
	- Include unit test?

"""

#########################################################################################
# Imports
from DataFlow.histogramBuilder import HistogramBuilder

import numpy as np
import os



#########################################################################################
# Function for the numerical toy case.
def setHistoBuilder_toyCase():
	"""
	Def.: Small geometry case for quick test histogramId and histogram.
	"""	
	histoBuilder = HistogramBuilder()
	nbDetLayer = 2
	nbDetAxial = 4
	nbDetRadial = 8
	maxAxialDiff = 2
	# Just want a fan of three detectors
	maxRadialCoverageRatio = 0.0001
	histoBuilder.nbDetBufferInRad = 1
	histoBuilder.setHistogramConfig(nbDetLayer, nbDetAxial, nbDetRadial, maxAxialDiff, \
										maxRadialCoverageRatio)
	
	return histoBuilder


def genDataTest_toyCase(histoId, _listModePath):
	"""
	Def.: Create data for the toy case.
	"""	
	randomProj = (5, 15, 489, 6, 448, 6, 187, 554, 15, 554, 55, 10 ,288,55, 55, 649, 253)
	
	# Note: the three that are added are not in histogramId. Thus, should not be in 
	# the histogram.
	listMode = np.vstack((histoId[randomProj, :], np.array([[0, 1], [60, 61], [0, 28]])))

	listMode.astype('int32').tofile(_listModePath)
							
	histogramComp = np.zeros(672)
	pos, counts = np.unique(randomProj, return_counts=True)
	for i, cPos in enumerate(pos):
		histogramComp[cPos] = counts[i]
							
	return histogramComp


def compareHistoId_toyCase():
	"""
	Def.: Compare the histogram Id created with the toy case using the ram and on 
		disk mode.
	"""	
	histoBuilder = setHistoBuilder_toyCase()
	
	# Load the true histogram Id.
	histoIdPathComp = "test/comp_toyCase_HistoId.dat"
	nbProj = int(os.path.getsize(histoIdPathComp) / (2 * (np.dtype(np.int32)).itemsize))
	histoIdComp = np.memmap(histoIdPathComp, dtype='int32', mode='r', shape=(nbProj, 2))	
	
	createdHistoIdFilePath = "test/tmp_histoId.dat"
	
	# Test ram mode. Delete the output file, if it already exist, to make sure we don't 
	# load an older file.
	if os.path.exists(createdHistoIdFilePath):
		os.remove(createdHistoIdFilePath)
	histoBuilder.genHistogramId(createdHistoIdFilePath, "ram")
	print("Compare histoId with ram mode.")
	print(np.all(histoBuilder.histogramId == histoIdComp))
	if os.path.exists(createdHistoIdFilePath):
		os.remove(createdHistoIdFilePath)	
	
	# Test disk mode. Delete the output file, if it already exist, to make sure we don't 
	# load an older file.
	if os.path.exists(createdHistoIdFilePath):
		os.remove(createdHistoIdFilePath)	
	histoBuilder.genHistogramId(createdHistoIdFilePath, "onDisk")
	print("Compare histoId with on disk mode.")
	print(np.all(histoBuilder.histogramId == histoIdComp))
	if os.path.exists(createdHistoIdFilePath):
		os.remove(createdHistoIdFilePath)	


def compareHistogram_toyCase():
	""" 
	Def.: Compare the histogram created with the toy case using the ram and on 
		disk mode, with parallel or not.
	"""	
	histoBuilder = setHistoBuilder_toyCase()
	
	listModePath = 'test/listMode_toyCase.dat'
	histoIdPath = "test/comp_toyCase_HistoId.dat"
	nbProj = int(os.path.getsize(histoIdPath) / (2 * (np.dtype(np.int32)).itemsize))
	histoId = np.memmap(histoIdPath, dtype='int32', mode='r', shape=(nbProj, 2))	
	histogramComp = genDataTest_toyCase(histoId, listModePath)
	
	# Set the path for the test.
	createdHistPath = 'test/test_histogram.dat'
	
	# Test ram mode. Delete the output file, if it already exist, to make sure we don't 
	# load an older file.
	if os.path.exists(createdHistPath):
		os.remove(createdHistPath)		
	histoBuilder.convertListModeToHistogram(listModePath, createdHistPath, \
												histoIdPath, _cMode='ram', _nbProcess=1)
	print("Compare histogram with on ram mode, not parallel.")
	print(np.all(histogramComp == histoBuilder.histogram))
	if os.path.exists(createdHistPath):
		os.remove(createdHistPath)	
	
	# Test disk mode. Delete the output file, if it already exist, to make sure we don't 
	# load an older file.	
	if os.path.exists(createdHistPath):
		os.remove(createdHistPath)	
	histoBuilder.convertListModeToHistogram(listModePath, createdHistPath, \
											histoIdPath, _cMode='onDisk', _nbProcess=1)
	print("Compare histogram with on disk mode, not parallel.")
	print(np.all(histogramComp == histoBuilder.histogram))
	
	
	# Test ram mode with dictionnary. Delete the output file, if it already exist, to  
	# make sure we don't load an older file.	
	if os.path.exists(createdHistPath):
		os.remove(createdHistPath)	
	histoBuilder.convertListModeToHistogramDict(listModePath, createdHistPath, \
											histoIdPath, _cMode='ram', _nbProcess=1)
	print("Compare histogram with on ram mode using dictionnary, not parallel.")
	print(np.all(histogramComp == histoBuilder.histogram))		
	
	# Test disk mode with dictionnary. Delete the output file, if it already exist, to  
	# make sure we don't load an older file.	
	if os.path.exists(createdHistPath):
		os.remove(createdHistPath)	
	histoBuilder.convertListModeToHistogramDict(listModePath, createdHistPath, \
											histoIdPath, _cMode='onDisk', _nbProcess=1)
	print("Compare histogram with on disk mode using dictionnary, not parallel.")
	print(np.all(histogramComp == histoBuilder.histogram))	
	
	
	# Test ram mode with Parallel. Delete the output file, if it already exist, to make 
	# sure we don't load an older file.	
	if os.path.exists(createdHistPath):
		os.remove(createdHistPath)	
	histoBuilder.nbProcToLaunch = 8
	histoBuilder.convertListModeToHistogram(listModePath, createdHistPath, \
												histoIdPath, _cMode='ram', _nbProcess=6)
	print("Compare histogram with on ram mode, parallel.")
	print(np.all(histogramComp == histoBuilder.histogram))
	
	# Test on disk mode with Parallel. Delete the output file, if it already exist, to  
	# make sure we don't load an older file.	
	if os.path.exists(createdHistPath):
		os.remove(createdHistPath)	
	histoBuilder.nbProcToLaunch = 8
	histoBuilder.convertListModeToHistogram(listModePath, createdHistPath, \
											histoIdPath, _cMode='onDisk', _nbProcess=6)
	print("Compare histogram with on ram mode, parallel.")
	print(np.all(histogramComp == histoBuilder.histogram))	



#########################################################################################
# Function for the temporary true but minimal case.

def setHistoBuilder_trueMinimalCase():
	"""
	Def.: Set the scanner configuration of the true minimal case.
	"""	
	histoBuilder = HistogramBuilder()
	histoBuilder.setHistogramConfigToSavantDefault()
	histoBuilder.maxAxialDiff = 1
	
	return histoBuilder


def compareHistoId_trueMinimalCase():
	"""
	Def.: Compare the histogram Id created with a true minimal case using the ram and on 
		disk mode.
	"""	
	histoBuilder = setHistoBuilder_trueMinimalCase()
	
	# Load the true histogram Id.
	histoIdPathComp = "test/comp_HistoId.dat"
	nbProj = int(os.path.getsize(histoIdPathComp) / (2 * (np.dtype(np.int32)).itemsize))
	histoIdComp = np.memmap(histoIdPathComp, dtype='int32', mode='r', shape=(nbProj, 2))	
	
	createdHistoIdFilePath = "test/tmp_histoId.dat"
	
	# Test ram mode. Delete the output file, if it already exist, to make sure we don't 
	# load an older file.
	if os.path.exists(createdHistoIdFilePath):
		os.remove(createdHistoIdFilePath)
	histoBuilder.genHistogramId(createdHistoIdFilePath, "ram")
	print("Compare histoId with ram mode.")
	print(np.all(histoBuilder.histogramId == histoIdComp))
	if os.path.exists(createdHistoIdFilePath):
		os.remove(createdHistoIdFilePath)	
	
	# Test disk mode. Delete the output file, if it already exist, to make sure we don't 
	# load an older file.
	if os.path.exists(createdHistoIdFilePath):
		os.remove(createdHistoIdFilePath)	
	histoBuilder.genHistogramId(createdHistoIdFilePath, "onDisk")
	print("Compare histoId with on disk mode.")
	print(np.all(histoBuilder.histogramId == histoIdComp))
	if os.path.exists(createdHistoIdFilePath):
		os.remove(createdHistoIdFilePath)	
	

def compareHistogram_trueMinimalCase():
	"""
	Def.: Compare the histogram created with a true minimal case using the ram and on 
		disk mode, with parallel or not.
	"""	
	histoBuilder = setHistoBuilder_trueMinimalCase()
	
	# Set the path for the test.
	histoIdPath = "test/comp_HistoId.dat"
	listModePath = 'test/listMode.dat'
	histoPathComp = 'test/comp_histogram.dat'
	histoComp = np.memmap(histoPathComp, dtype='float32', mode='r')	
	
	createdHistPath = 'test/test_histogram.dat'
	
	# Test ram mode. Delete the output file, if it already exist, to make sure we don't 
	# load an older file.
	if os.path.exists(createdHistPath):
		os.remove(createdHistPath)		
	histoBuilder.convertListModeToHistogram(listModePath, createdHistPath, \
												histoIdPath, _cMode='ram')
	print("Compare histogram with on ram mode, not parallel.")
	print(np.all(histoComp == histoBuilder.histogram))
	if os.path.exists(createdHistPath):
		os.remove(createdHistPath)	
	
	# Test ram mode with the dictionnary approach. Delete the output file, if it already 
	# exist, to make sure we don't  load an older file.
	if os.path.exists(createdHistPath):
		os.remove(createdHistPath)		
	histoBuilder.convertListModeToHistogramDict(listModePath, createdHistPath, \
												histoIdPath, _cMode='ram')
	print("Compare histogram with on ram mode using dictionnary, not parallel.")
	print(np.all(histoComp == histoBuilder.histogram))
	if os.path.exists(createdHistPath):
		os.remove(createdHistPath)	

	# Test disk mode. Delete the output file, if it already exist, to make sure we don't 
	# load an older file.	
	if os.path.exists(createdHistPath):
		os.remove(createdHistPath)	
	histoBuilder.convertListModeToHistogram(listModePath, createdHistPath, \
											histoIdPath, _cMode='onDisk',  _nbProcess=4)
	print("Compare histogram with on ram mode, parallel.")
	print(np.all(histoComp == histoBuilder.histogram))
	
	# Test ram mode with Parallel. Delete the output file, if it already exist, to make 
	# sure we don't load an older file.	
	if os.path.exists(createdHistPath):
		os.remove(createdHistPath)	
	histoBuilder.nbProcToLaunch = 8
	histoBuilder.vMode=2
	histoBuilder.convertListModeToHistogram(listModePath, createdHistPath, \
											histoIdPath, _cMode='ram', _nbProcess=4)
	print("Compare histogram with on ram mode, not parallel.")
	print(np.all(histoComp == histoBuilder.histogram))
	

#########################################################################################
# Old testing function, kept for future incorporation
def testHisoIdExtremeCases(_histoIdPath):
	'''
	Def.: Check if some projections are in the histogram Id. Some of the projections  
		chosen should be in the histogram Id but other should be excluded.
	Note: Not cleaned.
	'''
	dataType = np.dtype([('d1',np.int32), ('d2',np.int32)])
	nbProj = int(os.path.getsize(_histoIdPath) / (dataType.itemsize))
	histoId = np.memmap(_histoIdPath, dtype='int32', mode='r', shape=(nbProj, 2))

	totalNbElementDiffLayer = 377216
	totalNbElementSameLayer = 188608
	nbProjPerRingSameRing = totalNbElementDiffLayer + 2 * totalNbElementSameLayer
	nbProjPerRingDiffRing = 4 * totalNbElementDiffLayer
	nbLayer = 2
	nbAx = 144
	nbRad = 896
	nbRingDiff = 24
	nbLayerComb = 3	
	nbElementPerRingDiff = np.zeros(nbRingDiff + 2, dtype='int64')
	nbElementPerRingDiff[1] = nbProjPerRingSameRing * nbAx
	nbElementPerRingDiff[2:] = nbProjPerRingDiffRing \
									* (nbAx - np.arange(1, nbRingDiff + 1))	
	nbElementPerRingDiff = np.cumsum(nbElementPerRingDiff)

	# Fow now, lets test only middle cases and one not working
	dTestBasicSucc = np.array([0, 448])
	dTestBasicFail = np.array([400, 448])
	dTestOnlyUniqueWithDiffLayerOrRing = np.array([600, 152])
	testList = [[dTestBasicSucc, True, True], [dTestBasicFail, False, False], \
					 [dTestOnlyUniqueWithDiffLayerOrRing, False, True]]
	for testCase in testList:
		projId = testCase[0]
		print("Testing:" + str(projId))
		# Included, Layer 0-0, Ring 0-0, cRing 0
		bIncluded = testCase[1]
		checkIfDetectorIdOkay(projId, bIncluded, \
							[nbElementPerRingDiff[0], nbElementPerRingDiff[1]], histoId)
		# Included, Layer 0-0, Ring 10-10, cRing 0
		bIncluded = testCase[1]
		checkIfDetectorIdOkay(projId + 10 * 896, bIncluded, \
							[nbElementPerRingDiff[0], nbElementPerRingDiff[1]], histoId)
		# Included, Layer 0-1, Ring 0-0, cRing 0
		bIncluded = testCase[2]
		checkIfDetectorIdOkay(projId + np.array([0, 144 * 896]), bIncluded, \
							[nbElementPerRingDiff[0], nbElementPerRingDiff[1]], histoId)
		# Included, Layer 0-1, Ring 10-10, cRing 0
		bIncluded = testCase[2]
		checkIfDetectorIdOkay(projId + np.array([0, 144 * 896]) + (10 * 896), \
				bIncluded, [nbElementPerRingDiff[0], nbElementPerRingDiff[1]], histoId)
		# Included, Layer 1-1, Ring 0-0, cRing 0
		bIncluded = testCase[1]
		checkIfDetectorIdOkay(projId + (144 * 896), bIncluded, \
							[nbElementPerRingDiff[0], nbElementPerRingDiff[1]], histoId)
		# Included, Layer 1-1, Ring 10-10, cRing 0
		bIncluded = testCase[1]
		checkIfDetectorIdOkay(projId + (144 * 896) + (10 * 896), bIncluded, \
							[nbElementPerRingDiff[0], nbElementPerRingDiff[1]], histoId)
								
		bIncluded = testCase[2]
		# Included, Layer 0-0, Ring 0-0, cRing 5
		checkIfDetectorIdOkay(projId + np.array([0, 5 * 896]), bIncluded, \
							[nbElementPerRingDiff[5], nbElementPerRingDiff[6]], histoId)
		
		# Included, Layer 0-0, Ring 10-10, cRing 5
		checkIfDetectorIdOkay(projId + 10 * 896 + np.array([0, 5 * 896]), bIncluded, \
							[nbElementPerRingDiff[5], nbElementPerRingDiff[6]], histoId)
		
		# Included, Layer 0-1, Ring 0-0, cRing 5
		checkIfDetectorIdOkay(projId + np.array([0, 144 * 896]) \
									+ np.array([0, 5 * 896]), bIncluded, \
							[nbElementPerRingDiff[5], nbElementPerRingDiff[6]], histoId)
		
		# Included, Layer 0-1, Ring 10-10, cRing 5
		checkIfDetectorIdOkay(projId + np.array([0, 144 * 896]) + (10 * 896) \
									+ np.array([0, 5 * 896]), bIncluded, \
							[nbElementPerRingDiff[5], nbElementPerRingDiff[6]], histoId)
								
		# Included, Layer 1-1, Ring 0-0, cRing 5
		checkIfDetectorIdOkay(projId + (144 * 896) + np.array([0, 5 * 896]), bIncluded, \
							[nbElementPerRingDiff[5], nbElementPerRingDiff[6]], histoId)
		
		# Included, Layer 1-1, Ring 10-10, cRing 5
		checkIfDetectorIdOkay(projId + (144 * 896) + (10 * 896) \
									+ np.array([0, 5 * 896]), bIncluded, \
							[nbElementPerRingDiff[5], nbElementPerRingDiff[6]], histoId)
		# todo: 1-0

	# Case that did not work before
	# Case where failed to see that all combination of radial are valid for layer 0-1
	# combinaison 
	checkIfDetectorIdOkay(np.array([3432, 132206]), True, \
							[nbElementPerRingDiff[0], nbElementPerRingDiff[1]], histoId)
	# Dunno why I checked this one by hand
	checkIfDetectorIdOkay(np.array([20251, 20484]), False, \
							[nbElementPerRingDiff[0], nbElementPerRingDiff[0]], histoId)
	# Negative ring difference
	checkIfDetectorIdOkay(np.array([104755, 231574]), True, \
							[nbElementPerRingDiff[2], nbElementPerRingDiff[3]], histoId)	



def checkIfDetectorIdOkay(_detId, _bIncluded, _idRange, _histoId):
	'''
	Def.: Simple function that check if the projection was included or not and if that 
		was okay.
	Note: Not cleaned.
	'''
	firstCheckPos = np.where(_histoId[_idRange[0]:_idRange[1], 0] == _detId[0])[0]
	if _bIncluded and (firstCheckPos.size == 0):
		print("Premptive fail at " + str(_detId) + " should have been included.")
	secondCheckPos = np.where(_histoId[_idRange[0]:_idRange[1], :][firstCheckPos, 1] \
								== _detId[1])[0]
	if _bIncluded and (secondCheckPos.size == 0):
		print("Fail at " + str(_detId) + " should have been included.")
	if _bIncluded and (secondCheckPos.size > 1):
		print("Fail at " + str(_detId) + ". More than one possibilites detected.")
	if (not _bIncluded) and (secondCheckPos.size > 0):
		print("Fail at " + str(_detId) + ". should not have been included.")

def testOverallHisoId():
	'''
	Def.: Check if the projections respect the expected block pattern.
	Note: Not cleaned.
	'''	
	# Param
	totalNbElementDiffLayer = 377216
	totalNbElementSameLayer = 188608
	nbProjPerRingSameRing = totalNbElementDiffLayer + 2 * totalNbElementSameLayer
	nbProjPerRingDiffRing = 4 * totalNbElementDiffLayer
	nbLayer = 2
	nbAx = 144
	nbRad = 896
	nbRingDiff = 24
	nbLayerComb = 3

	# Interm. value
	nbElementPerRingDiffV2 = np.zeros(nbRingDiff + 2, dtype='int64')
	nbElementPerRingDiffV2[1] = nbProjPerRingSameRing * nbAx
	nbElementPerRingDiffV2[2:] = nbProjPerRingDiffRing \
									* (nbAx - np.arange(1, nbRingDiff + 1))	
	nbElementPerRingDiffV2 = np.cumsum(nbElementPerRingDiffV2)

	for cRingDiff in range(nbRingDiff +1):
		# Validate cring
		extStartId = nbElementPerRingDiffV2[cRingDiff]
		extEndId = nbElementPerRingDiffV2[cRingDiff+1]
		tmpHistoId = np.array(histoId[extStartId:extEndId, :])
		startId = 0
		endId = extEndId - extStartId
		valid = np.all(np.diff((tmpHistoId[startId:endId, :] % (144*896)) // 896) \
						== cRingDiff)
		if not valid:
			print("Ring diff " + str(cRingDiff) + " is not valid")
		for cRing in range(1, nbAx - cRingDiff):
			if cRingDiff == 0:
				startId = cRing * nbProjPerRingSameRing
				endId = startId + nbProjPerRingSameRing
			else:
				startId = cRing * nbProjPerRingDiffRing
				endId = startId + nbProjPerRingDiffRing
			valid = np.all((tmpHistoId[startId:endId, 0] % (144*896)) // 896 == cRing)
			if not valid:
				print("Ring " + str(cRing) + " is not valid")
		print("Ring diff " + str(cRingDiff) + " is done.")
		
		
		
		
#########################################################################################
# Main : We use this to make the main usable has a script and a method.
#########################################################################################

if __name__== "__main__":
	"""
	Currently, this is used to make test by hand.
	"""
	# compareHistoId_trueMinimalCase()
	# compareHistogram_trueMinimalCase()
	# compareHistoId_toyCase()
	compareHistogram_toyCase()
	