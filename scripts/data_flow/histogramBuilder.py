#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module used to create histograms. Currently, the work flow is that the Class 
HistogramBuilder hold all the variables needed and process them to create the histogram.
The main is used for the basic case of building the histogram for the Savant DOI from a 
Gate binary file.

Since the histogram bins that are empty are kept for now, the projection detector Id for 
a given Field-of-View configuration can be reused with any simulation made. 

TODO:
	- Setup the main for easy scripting.
	- Add a histogram version with time axis?
	- Add projection angle ordering?
	- Add save base block for histogram compression 
	- Add progress relative the number of events placed in the histogram?
	
Possible improvements:
	- basicRadialProjDetInSameRing and basicRadialProjDetInDiffRing could be used to 
		limit further the search when building the histogram. 
"""

#########################################################################################
# Imports
import sys
import numpy as np
import argparse
import os

from numba import jit, njit
import multiprocessing as mp
from multiprocessing import Array
from numba import types
from numba.typed import Dict


#########################################################################################
# Class of this module.
class HistogramBuilder:
	'''
	Object that hold the parameters and tools needed to convert a listMode to a 
	histogram.
	
	List of features:
		- Build basic histogram.
			- Can use radial and axial condition to limit the projection space field of
				view.
		- Convert Gate to histogram.
		- Convert CASToR ListMode to histogram.

	'''

	def __init__(self):
		'''
		Def.: Explicit the differents recurring variables created/used in this class.
		'''
		# Basic configuration of the scanner.
		self.nbDetLayer = -1
		self.nbDetAxial = -1
		self.nbDetRadial = -1
		# Basic configuration of Field-of-View.
		self.maxAxialDiff = -1
		self.maxRadialCoverageRatio = -1
		
		# Supplementary information about the scanner configuration.
		self.lutFile = ""
		
		# Intermediate definition of the scanner.
		# Currently not stored but could be used to accelerate further search in the 
		# histogram.
		# self.basicRadialProjDetInSameRing = -1
		# self.basicRadialProjDetInDiffRing = -1
		
		# Variable that can be created by this object.
		self.histogramId = None
		self.histogram = None
		
		# Constants. Values that there should be no need to change.
		# Number of detector we add as a buffer to the one obtained from making a fan 
		# from a given radial position.
		self.nbDetBufferInRad = 4
		
		# Parameters used for parallel processing.
		self.nbProcToLaunch = 1
		# '/home/MToussaint/wakusuteshon/2019_03_29_validateYoannProj/test/'
		self.pathForMultiProcTempFile = ""
		
		# Level of verbosity.
		self.vMode = 0
		
		
	def setHistogramConfig(self, _nbDetLayer, _nbDetAxial, _nbDetRadial, \
							_maxAxialDiff=-1, _maxRadialCoverageRatio=-1, _lutFile = ""):
		'''
		Def.: Set the configuration of the histogram to build.
		'''
		self.nbDetLayer = _nbDetLayer
		self.nbDetAxial = _nbDetAxial
		self.nbDetRadial = _nbDetRadial
		self.maxAxialDiff = _maxAxialDiff
		self.maxRadialCoverageRatio = _maxRadialCoverageRatio
		self.lutFile = _lutFile
		

	def setHistogramConfigToSavantDefault(self):
		'''
		Def.: Set the configuration of the Savant DOI scanner. It also set the 
			projection space Field of View to the current maxium considered.
		'''
		self.nbDetLayer = 2
		self.nbDetAxial = 144
		self.nbDetRadial = 896
		# Currently, we limit the axial field of view to a third of the scanner.		
		self.maxAxialDiff = 24
		# Currently, the radial field of view radius is limited to two third the 
		# diameter of the scanner and defined as a cylinder.
		self.maxRadialCoverageRatio = np.arcsin((26/2)/(39/2)) * 2.0 / np.pi


	def genHistogramId(self, _outputFile, _cMode, _oMode=""):
		'''
		Def.: Create and save the histogram projection detector Id following the 
			pre-setted scanner configuration.
		Note: Here, a ring, is defined as the subset of detectors pair for a given 
			axial-s combination. 
		@_outputFile (String): Patht to the file where the histogram detector Id will be
			saved.
		@_cMode (String): Indicate if the file should be kept in ram and then saved or 
			if it should be directly written on disk.
		@_oMode (String): Define the type of ordeing to use for projection block.
		'''
		
		# Valid projections when only the radial Id vary. 
		radialBasisProjDetId_sameRing, radialBasisProjDetId_diffRing \
															= self.genRadProjDetIdBasis()
		
		# Re-ordering following the chosen mode.
		radialBasisProjDetId_sameRing, radialBasisProjDetId_diffRing \
						= self.radialBasisReOrdering(radialBasisProjDetId_sameRing, \
													radialBasisProjDetId_diffRing,
													_oMode)		
					
		# Compute number of unique projection for the current configuration.
		# Include layer combination dimension.
		nbProjPerAxialPos_SameAxialId = (2 * radialBasisProjDetId_sameRing.shape[0]) \
								+ (radialBasisProjDetId_diffRing.shape[0])
		nbProjPerAxialPos_DiffAxialId = 4 * radialBasisProjDetId_diffRing.shape[0]
		# Include axial and axial difference dimension.
		nbProjUnique = self.nbDetAxial * nbProjPerAxialPos_SameAxialId \
			+ nbProjPerAxialPos_DiffAxialId \
				* np.arange(self.nbDetAxial - self.maxAxialDiff, self.nbDetAxial).sum()	
										
		self.initializeHistogramId(_outputFile, nbProjUnique, _cMode)
			
		# The projection are added in block of axial Id and then in (positive) 
		# difference of axial Id. Each block is built from the basis of valid 
		# projections obtained when only radial Id vary. This is seperated in two cases:
		# those that share the same axial Id and those that does not.
		startIndexCurrAx = 0 
		for axialDiffId in range(self.maxAxialDiff + 1):	
			if self.vMode > 0:
				print("Ring difference: " + str(axialDiffId))
				
			for axId in range(self.nbDetAxial - axialDiffId):
				if self.vMode > 1:
					print("  Axial: " + str(axId))
				if axialDiffId == 0:
					self.fillHistoIdPart_sameAxialId(radialBasisProjDetId_sameRing, \
													radialBasisProjDetId_diffRing, \
													startIndexCurrAx, axId)
					startIndexCurrAx += nbProjPerAxialPos_SameAxialId
				else:
					self.fillHistoIdPart_diffAxialId(radialBasisProjDetId_diffRing, \
													startIndexCurrAx, axId, axialDiffId)
					startIndexCurrAx += nbProjPerAxialPos_DiffAxialId	
					
		self.saveHistogramId(_outputFile, _cMode)	
	

	def initializeHistogramId(self, _outputFile, _nbProjUnique, _cMode):
		'''
		Def.: Initialize the histogram Id following the defined computation mode.
		@_outputFile (String): Patht to the file where the histogram detector Id will be
			saved.
		@_nbProjUnique (Integer): Total number of unique projection in the histogram.
		@_cMode (String): Indicate if the file should be constructed in ram and than 
			saved or if it should be directly written on disk.
		'''
		# Since all the elements of histoId will be overwritten, we do not need to 
		# initialize its values.
		if _cMode == "ram":
			self.histogramId = np.empty((_nbProjUnique, 2), dtype='int32')
		elif _cMode == "onDisk":
			self.histogramId = np.memmap(_outputFile, dtype='int32', mode='w+', \
											shape=(_nbProjUnique, 2))
		else:
			print("The mode " + str(_cMode) + " is not avaialble. Exiting.")
			sys.exit(1)

		
	def genRadProjDetIdBasis(self):
		""" 
		Def.: Generate the detector Id of valid projections for the Field of View 
			defined. Only the radial dimension is considered. Two arrays are produced: 
			one for the case of projections defined by detector Id on the same axial and 
			layer dimension and another for the other case. We differ these cases as 
			'same ring' vs 'different ring'.
		Note: Most likely only works since we are in a scanner that is similar to a 
			cylinder.
		Return:
		 	radialBasisProjDetId_sameRing (2D numpy, Interger)
			radialBasisProjDetId_diffRing (2D numpy, Interger)
		TODO: Split? See dfg
		"""
		# dfg This part could get fancy-er and should be done in an another method.
		# Get the detector Id that make a valid projection with the detector of radial 
		# Id 0.
		if self.maxRadialCoverageRatio != -1:
			# Use the maximum radial coverage to get the maximum number of valid  
			# projection possible if one of the detector Id is fixed.
			nbProjValidFromOneRadPos = int(\
							np.ceil(self.nbDetRadial * self.maxRadialCoverageRatio) \
							+ self.nbDetBufferInRad)

			smallestDetIdForDetRad0 = int(np.ceil(self.nbDetRadial / 2) \
												- np.floor(nbProjValidFromOneRadPos / 2))
			highestDetIdForDetRad0 = int(np.ceil(self.nbDetRadial / 2) \
												+ np.floor(nbProjValidFromOneRadPos / 2))
		else:
			smallestDetIdForDetRad0 = 0
			highestDetIdForDetRad0 = self.nbDetRadial
			
		validDetForCoincWithDetRad0 = np.arange(smallestDetIdForDetRad0, \
													highestDetIdForDetRad0 + 1)

		# For each radial Id, the same basic fan, centered at its opposite detector 
		# should be a valid projection. After the highest Id of the case radial 0, the 
		# projections have already been considered previously.
		baseRadProjDetId_SameRing = validDetForCoincWithDetRad0[np.newaxis, :] \
									+ np.arange(highestDetIdForDetRad0)[:, np.newaxis]
		# Any detector Id higher than the number of detector in the radial dimension 
		# where already considered since we started at detector Id 0 so we remove them.
		baseRadProjDetId_SameRing[\
						np.where(baseRadProjDetId_SameRing >= self.nbDetRadial)] = -1
		
		# For each radial Id, the same basic fan, centered at its opposite detector 
		# should be a valid projection.
		baseRadProjDetId_DiffRing = validDetForCoincWithDetRad0[np.newaxis, :] \
										+ np.arange(self.nbDetRadial)[:, np.newaxis] 
		# Since the detector in coincidence are not in the same axial and/or layer 
		# dimension, they are all unique. 
		baseRadProjDetId_DiffRing = baseRadProjDetId_DiffRing % self.nbDetRadial
		
		# dfg (From here, everything is duplicated for no reason)
		# From here, we create explicitly the Id1 of each projection and match them with
		# their Id2.
		nbProjRad_SameRing = (baseRadProjDetId_SameRing !=-1).sum(axis=1)
		nbProjRad_DiffRing = (baseRadProjDetId_DiffRing !=-1).sum(axis=1)	
		
		radialBasisDetId1_sameRing = np.repeat(\
					np.arange(baseRadProjDetId_SameRing.shape[0]), nbProjRad_SameRing)
		radialBasisDetId1_diffRing = np.repeat(\
					np.arange(baseRadProjDetId_DiffRing.shape[0]), nbProjRad_DiffRing)
		
		radialBasisDetId2_sameRing = baseRadProjDetId_SameRing[\
											np.where(baseRadProjDetId_SameRing != -1)]
		radialBasisDetId2_diffRing = baseRadProjDetId_DiffRing[\
											np.where(baseRadProjDetId_DiffRing != -1)]
		
		radialBasisProjDetId_sameRing = np.hstack(\
										(radialBasisDetId1_sameRing[:, np.newaxis], \
											radialBasisDetId2_sameRing[:, np.newaxis]))
		radialBasisProjDetId_diffRing = np.hstack(\
										(radialBasisDetId1_diffRing[:, np.newaxis], \
											radialBasisDetId2_diffRing[:, np.newaxis]))		
		
		return radialBasisProjDetId_sameRing, radialBasisProjDetId_diffRing



	def fillHistoIdPart_sameAxialId(self, _radialBasisProjDetId_sameRing, \
									_radialBasisProjDetId_diffRing, _startIndexCurrAx, \
									_axialId):
		"""
		Def.: Compute the detector Id of the valid projection for the current axial Id.
		@_radialBasisProjDetId_sameRing (2D numpy array, Integer): The radial Id of 
			all valid projections with detector Id pair that share the same ring. 
		@_radialBasisProjDetId_diffRing (2D numpy array, Integer): The radial Id of 
			all valid projections with detector Id pair that are not on the same ring. 
		@_startIndexCurrAx (Integer): The starting index of the part to fill.
		@_axialId (Integer): The axial Id of the part to fill.
		"""
		
		nbDetPerLayer = self.nbDetRadial * self.nbDetAxial

		# layer 0 with layer 0
		# The detector are on the same ring.
		firstProjIndex = _startIndexCurrAx
		lastProjIndex = firstProjIndex + _radialBasisProjDetId_sameRing.shape[0]
		self.histogramId[firstProjIndex:lastProjIndex, 0] = \
												_radialBasisProjDetId_sameRing[:, 0] \
												+ _axialId * self.nbDetRadial
		self.histogramId[firstProjIndex:lastProjIndex, 1] = \
												_radialBasisProjDetId_sameRing[:, 1] \
												+ _axialId * self.nbDetRadial
	
		# layer 0 with layer 1
		# The detector are NOT on the same ring. Also, the case layer 1 with layer 0 is 
		# redondant with this one.
		firstProjIndex = lastProjIndex
		lastProjIndex = firstProjIndex + _radialBasisProjDetId_diffRing.shape[0]
		self.histogramId[firstProjIndex:lastProjIndex, 0] = \
												_radialBasisProjDetId_diffRing[:, 0] \
												+ _axialId * self.nbDetRadial
		self.histogramId[firstProjIndex:lastProjIndex, 1] = \
												_radialBasisProjDetId_diffRing[:, 1] \
												+ _axialId * self.nbDetRadial \
												+ nbDetPerLayer
	
		# layer 1 with layer 1
		# The detector are on the same ring.
		firstProjIndex = lastProjIndex
		lastProjIndex = firstProjIndex + _radialBasisProjDetId_sameRing.shape[0]
		self.histogramId[firstProjIndex:lastProjIndex, 0] = \
												_radialBasisProjDetId_sameRing[:, 0] \
												+ _axialId * self.nbDetRadial \
												+ nbDetPerLayer 
		self.histogramId[firstProjIndex:lastProjIndex, 1] = \
												_radialBasisProjDetId_sameRing[:, 1] \
												+ _axialId * self.nbDetRadial \
												+ nbDetPerLayer


	def fillHistoIdPart_diffAxialId(self, _radialBasisProjDetId_diffRing, \
									_startIndexCurrAx, _axId, _axialDiffId):
		"""
		Def.: Compute the detector Id of the valid projection for the current lowest 
			axial Id and current difference in axial Id.
		@_radialBasisProjDetId_sameRing (2D numpy array, Integer): The radial Id of 
			all valid projections with detector Id pair that are not on the same ring. 
		@_startIndexCurrAx (Integer): The starting index of the part to fill.
		@_axialId (Integer): The axial Id of the part to fill.
		@_axialId (Integer): The positive difference in axial Id of the part to fill.
		"""
		nbDetPerLayer = self.nbDetRadial * self.nbDetAxial

		# layer 0 with layer 0
		# The detector are NOT on the same ring.
		firstProjIndex = _startIndexCurrAx
		lastProjIndex = firstProjIndex + _radialBasisProjDetId_diffRing.shape[0]
		self.histogramId[firstProjIndex:lastProjIndex, 0] = \
											_radialBasisProjDetId_diffRing[:, 0] \
											+ _axId * self.nbDetRadial
		self.histogramId[firstProjIndex:lastProjIndex, 1] = \
											_radialBasisProjDetId_diffRing[:, 1] \
											+ (_axId + _axialDiffId) * self.nbDetRadial 

		# layer 0 with layer 1
		# The detector are NOT on the same ring.
		firstProjIndex = lastProjIndex
		lastProjIndex = firstProjIndex + _radialBasisProjDetId_diffRing.shape[0]
		self.histogramId[firstProjIndex:lastProjIndex, 0] = \
											_radialBasisProjDetId_diffRing[:, 0] \
											+ _axId * self.nbDetRadial
		self.histogramId[firstProjIndex:lastProjIndex, 1] = \
											_radialBasisProjDetId_diffRing[:, 1] \
											+ (_axId + _axialDiffId) * self.nbDetRadial \
											+ nbDetPerLayer

		# layer 1 with layer 0, negative axial difference.
		# The detector are NOT on the same ring.
		# The detector Id are saved as layer 0 than layer 1 since we want the detector 
		# Id 1 to be lower than the detector Id 2.
		firstProjIndex = lastProjIndex
		lastProjIndex = firstProjIndex + _radialBasisProjDetId_diffRing.shape[0]
		self.histogramId[firstProjIndex:lastProjIndex, 0] = \
											_radialBasisProjDetId_diffRing[:, 1] \
											+ (_axId + _axialDiffId) * self.nbDetRadial
		self.histogramId[firstProjIndex:lastProjIndex, 1] = \
											_radialBasisProjDetId_diffRing[:, 0] \
											+ _axId * self.nbDetRadial \
											+ nbDetPerLayer

		# layer 1 with layer 1
		# The detector are NOT on the same ring.
		firstProjIndex = lastProjIndex
		lastProjIndex = firstProjIndex + _radialBasisProjDetId_diffRing.shape[0]
		self.histogramId[firstProjIndex:lastProjIndex, 0] = \
											_radialBasisProjDetId_diffRing[:, 0] \
											+ _axId * self.nbDetRadial \
											+ nbDetPerLayer 
		self.histogramId[firstProjIndex:lastProjIndex, 1] = \
											_radialBasisProjDetId_diffRing[:, 1] \
											+ (_axId + _axialDiffId) * self.nbDetRadial \
											+ nbDetPerLayer


	def radialBasisReOrdering(self, _radialBasisProjDetId_sameRing, \
								_radialBasisProjDetId_diffRing, _oMode):
		"""
		Def.: Re-order the projections in both radial basis such that their projections
		 	are sorted following the specified mode.
		@_radialBasisProjDetId_sameRing (2D numpy array, Integer): The radial Id of 
			all valid projections with detector Id pair that share the same ring. 
		@_radialBasisProjDetId_diffRing (2D numpy array, Integer): The radial Id of 
			all valid projections with detector Id pair that are not on the same ring. 
		@_oMode (String): Type of re-ordering to use.
		Return:
			(2D numpy array, Integer)[-1, 2]
			(2D numpy array, Integer)[-1, 2]
		"""
		
		if _oMode == "":
			return _radialBasisProjDetId_sameRing, _radialBasisProjDetId_diffRing
		elif _oMode == "pureAngle":
			# Re-order the projection in both radial basis such that their projections
			# are sorted in ascending order of projection angle relative to (1, 0, 0).
			if self.lutFile == "":
				print("The lut of detector spatial position must be given to use the "\
						"pureAngle mode.")
			return self.pure2DAngleOrdering(_radialBasisProjDetId_sameRing), \
					self.pure2DAngleOrdering(_radialBasisProjDetId_diffRing)
		else:
			print("The radial basis ordering mode " + str(_oMode) + " is not defined.")


	def pure2DAngleOrdering(self, _projDetId):
		"""
		Def.: Re-order the projection in a radial basis in ascending order of their 
			angle of projection.
			Note: Work as intended but the basic case seems to work to so...
		@_projDetId (2D numpy array, Integer): Detector Id of a radial basis.
		Return:
			(2D numpy array, Integer)[-1, 2]
		"""
		
		# Only take the x, y positions of the detector.
		detPos2D = np.fromfile(self.lutFile, dtype=np.dtype(('f', 6)))[:, :2]

		proj2DVec = detPos2D[_projDetId[:, 0], :] - detPos2D[_projDetId[:, 1], :] 
		proj2DVecNorm = proj2DVec / np.linalg.norm(proj2DVec, axis=1)[:, np.newaxis]
		
		# We define the Beta with angle counter-clockwise to the positive x-axis. In 
		# that case, vector with negative y-axis would have an angle that is np.pi more 
		# than what is expected (Since we want the angle of the projection represented 
		# by that vector.) Thus, we invert the vector in those cases.
		proj2DVecNorm[np.where(proj2DVecNorm[:, 1] < 0.0)] *= -1.0
		beta = np.arccos(proj2DVecNorm[:, 0])
						
		return _projDetId[np.argsort(beta)]


	def saveHistogramId(self, _histoIdPath, _cMode):
		""" 
		Def.: Save the histogram Id at the specified path if it was not already created 
			on disk.
		@_histoIdPath (String): Path where the histogramId will be saved.
		@_cMode (String): Computation mode used. If the histogramId was kept in ram, we 
			need to write it to the file. 
		"""
		if _cMode == "ram":
			if self.vMode > 0:
				print("Saving the histoId on disk.")		
			self.histogramId.tofile(_histoIdPath)
		elif  _cMode == "onDisk":
			if self.vMode > 0:
				print("The histoId file is already written on disk.")	


	def convertListModeToHistogramDict(self, _listModePath, _histoPath, _histoIdPath, \
									_cMode="", _iMode="basic", _timeSlice=None, 
									_nbProcess=1):
		"""
		Def.: Convert a List mode dataset into an histogram using the given histogramId 
			using a dictionnary of the projection Id for the radial base.
		@_listModePath (String): Path to the file holding the data set in List Mode.
		@_histoPath (String): Path where the histogram will be saved.
		@_histoIdPath (String): Path to the histogram Id.
		@_cMode (String): Computation mode to use. Either "ram" or "onDisk". Currently,
		 	it is one or the other for all the variable.
		@_iMode (String): Type of List Mode saved in _listModePath.
		@_timeSlice dsa
		@_nbProcess (Integer): The number of process desired. (Might have more or less) 
		TODO:
			- Implement multiprocessing.
			- The dictionnary could be reused.
		""" 
		if _nbProcess != 1:
			print("Currently, multi processing is not implemented for this method.")
			# self.nbProcToLaunch = _nbProcess
		
		eventsDetId, eventsBlockId = self.extractEventDetIdAndBlockId(_listModePath, \
		                                                              _iMode, _timeSlice)
			
		self.loadHistogramId(_histoIdPath, _cMode)
		self.initHistogram(_histoPath, _cMode)

		firstProjIdPerBlock, supProjInBlock = self.genProjBlockPartition()
		eventProjId = self.initListModeProjIdTmpHolder(eventsDetId.shape[0])
		
		# Create a dictionnary of merged detector Id to projection Id for the 
		# projections with axials Id of 0 and those with an axial Id of 0 and 1.
		projIdBasisDict = self.genBasisProjIdDict(firstProjIdPerBlock)

		mergedDetId = self.genBasisMergedDetectorId(eventsDetId, eventsBlockId)
		
		findProjIdOfEventsDict(eventProjId, projIdBasisDict, eventsBlockId,  \
		                       mergedDetId, firstProjIdPerBlock, self.maxAxialDiff, \
							   self.vMode)	

		self.writeEventsInHistogram(eventProjId, _histoPath, _cMode)



	def convertListModeToHistogram(self, _listModePath, _histoPath, _histoIdPath, \
									_cMode="", _iMode="basic", _nbProcess=1):
		"""
		Def.: Convert a List mode dataset into an histogram using the given histogramId. 
		@_listModePath (String): Path to the file holding the data set in List Mode.
		@_histoPath (String): Path where the histogram will be saved.
		@_histoIdPath (String): Path to the histogram Id.
		@_cMode (String): Computation mode to use. Either "ram" or "onDisk". Currently,
		 	it is one or the other for all the variable.
		@_iMode (String): Type of List Mode saved in _listModePath.
		@_nbProcess (Integer): The number of process desired. (Might have more or less) 
		Note: I created findRelProjIdOfEventList to make this faster but it was a little 
			worst... Kind of weird since we should have less python interaction...
		"""
		self.nbProcToLaunch = _nbProcess
		
		eventsDetId, eventsBlockId = self.extractEventDetIdAndBlockId(\
																_listModePath, _iMode)
			
		self.loadHistogramId(_histoIdPath, _cMode)
		self.initHistogram(_histoPath, _cMode)

		firstProjIdPerBlock, supProjInBlock = self.genProjBlockPartition()
		eventProjId = self.initListModeProjIdTmpHolder(eventsDetId.shape[0])

		# Idea: The projection detector Id can be easily partiotionned in block of axial
		# Id and positive difference in axial Id. We do the same for the events thus 
		# allowing the search of the event projection Id to be limited to that block.
		for cAxialDiff in range(self.maxAxialDiff + 1):
			if self.vMode > 0:
				print("Current difference in axial Id: " + str(cAxialDiff))
			eventsListModePos_cAxialDiff = np.where(eventsBlockId[:, 2] == cAxialDiff)[0]
			# Find the first and last projection Id of the current axial difference.
			# Allow us to only load that part once.
			firstProjId_cAxialDiff = firstProjIdPerBlock[0, cAxialDiff]
			if cAxialDiff == self.maxAxialDiff:
				lastProjId_cAxialDiff = firstProjIdPerBlock[ \
											self.nbDetAxial - cAxialDiff, cAxialDiff]
			else:
				lastProjId_cAxialDiff = firstProjIdPerBlock[0, cAxialDiff + 1]
			
			# Load the current part if using disk mode since one axial difference block 
			# is not that big and reading a file is costly.
			if _cMode == "ram":
				subHistoId_cAxialDiff = self.histogramId[\
									firstProjId_cAxialDiff:lastProjId_cAxialDiff, :]
			elif _cMode == "onDisk":
				subHistoId_cAxialDiff = np.array(self.histogramId[\
									firstProjId_cAxialDiff:lastProjId_cAxialDiff, :])
				
			if self.nbProcToLaunch == 1:
				self.processCurrAxialDiff(eventProjId, cAxialDiff, \
									eventsBlockId, eventsListModePos_cAxialDiff, \
									firstProjIdPerBlock, firstProjId_cAxialDiff, \
									supProjInBlock, eventsDetId, subHistoId_cAxialDiff)
			else:
				self.processCurrAxialDiffParallel(eventProjId, cAxialDiff, \
									eventsBlockId, eventsListModePos_cAxialDiff, \
									firstProjIdPerBlock, firstProjId_cAxialDiff, \
									supProjInBlock, eventsDetId, subHistoId_cAxialDiff)

		self.writeEventsInHistogram(eventProjId, _histoPath, _cMode)



	def initListModeProjIdTmpHolder(self, _nbEvents):
		""" 
		Def.: Create an array that will hold the event projection Id. Mostly usefull 
			for the multi-process part to dodge lock complexity.
		@_nbEvents (Integer): The number of events in the ListMode.
		Return:
			(1D Numpy array or MultiProcessing array, Integer)[_nbEvents]
		"""
		# To limit IO back and forth, we save the event only at the end. Might not save 
		# much with the current version of the code. However, we can't use parallel 
		# easily if we write directly in the histogram.
		# Not sure if all events are valid for the current histrogram definition, so 
		# initialize everyone at -1.
		if self.nbProcToLaunch == 1:
			eventProjId = -np.ones(_nbEvents, dtype='int64')
		else:
			# Usefull when the tempdir of the system is to small (e.g. Meson)
			if self.pathForMultiProcTempFile != "" :
				mp.process.current_process()._config['tempdir'] \
														= self.pathForMultiProcTempFile

			eventProjId = Array('l', [-1]*_nbEvents, lock=False)
			
		return eventProjId
			
		
	def genBasisMergedDetectorId(self, _eventsDetId, _eventsBlockId):
		"""
		Def.: Generate the Merged detector Id of the events in their radial basis.
		@_eventsDetId (2D numpy array, Integer)[-1, 2]: The Id of the detector pair of 
			each events.
		@_eventsBlockId (2D numpy array, Integer)[-1, 3]: The block Id of each events.
		Return: 1D numpy array, Integer
		"""
		mergedDetId = np.zeros(_eventsDetId.shape[0], dtype='int64')
		sameAxialId = np.where(_eventsBlockId[:, 2] == 0)[0]
		diffAxialId = np.where(_eventsBlockId[:, 2] != 0)[0]
		
		# Explode without that.
		eDetId = _eventsDetId.astype('int64')
		# Compute the merged detector Id for event with detectors that share the same 
		# Axial Id. We re-defined them relative to Axial Id of 0 to correspond to the
		# cases saved in the dictionnary.
		mergedDetId[sameAxialId] = (eDetId[sameAxialId, 0] \
		                           - self.nbDetRadial * _eventsBlockId[sameAxialId, 1]) \
		                         + (eDetId[sameAxialId, 1] \
		                           - self.nbDetRadial * _eventsBlockId[sameAxialId, 1]) \
		                         * self.nbDetAxial * self.nbDetLayer * self.nbDetRadial
		# Compute the merged detector Id for event with detectors that don't share the  
		# same Axial Id. We re-defined them relative to Axial Id of 0 and difference of 
		# Axial Id of 1 to correspond to the cases saved in the dictionnary.	
		notFlip = np.where(_eventsBlockId[diffAxialId, 0] != 2)
		dAxId_nFlip = diffAxialId[notFlip]
		flip = np.where(_eventsBlockId[diffAxialId, 0] == 2)
		dAxId_flip = diffAxialId[flip]
		mergedDetId[dAxId_nFlip] = \
			                (eDetId[dAxId_nFlip, 0] \
			                    - self.nbDetRadial * _eventsBlockId[dAxId_nFlip, 1]) \
			                + (eDetId[dAxId_nFlip, 1] - self.nbDetRadial \
			                    * (_eventsBlockId[dAxId_nFlip, 1] \
			                           + _eventsBlockId[dAxId_nFlip, 2] - 1)) \
			                * self.nbDetAxial * self.nbDetLayer * self.nbDetRadial
		mergedDetId[dAxId_flip] = \
			        (eDetId[dAxId_flip, 0] \
			            - self.nbDetRadial * (_eventsBlockId[dAxId_flip, 1] \
			                                     + _eventsBlockId[dAxId_flip, 2] - 1))\
			        + (eDetId[dAxId_flip, 1] - self.nbDetRadial \
			            * _eventsBlockId[dAxId_flip, 1]) \
			        * self.nbDetAxial * self.nbDetLayer * self.nbDetRadial
			
		return mergedDetId
		
			
	def processCurrAxialDiff(self, _eventProjId, _cAxialDiff, _eventsBlockId, \
								_eventsListModePos_cAxialDiff, _firstProjIdPerBlock, \
								_firstProjId_cAxialDiff, _supProjInBlock, \
								_eventsDetId, _subHistoId_cAxialDiff):
		"""
		Def.: Find the projection Id for the current difference in Axial Id. 
		@_eventProjId (1D numpy array, Integer): The array that will contain the 
			projection Id of the events.
		@_cAxialDiff (Integer): The current axial difference.
		@_eventsBlockId (2D numpy array, Integer)[-1, 3]: The block Id of each events.
		@_eventsListModePos_cAxialDiff (1D numpy array, Integer): Position in the listMode 
			of the events that are in the current axial difference.
		@_firstProjIdPerBlock (2D numpy array, Integer)[nbDetAxail, maxAxialDiff-1]: The 
			projection Id of the first projection of each block of axial Id and 
			difference in axial Id. 
		@_firstProjId_cAxialDiff (Integer): The projection Id of the first element of 
			_subHistoId_cAxialDiff.
		@_supProjInBlock (Integer): An upper limit on the number of projections that 
			are in a given axial and axial difference block. Used to limit the search 
			when an event is not in the projection Field of View.
		@_eventsDetId (2D numpy array, Integer)[-1, 2]: The Id of the detector pair of 
			each events.
		@_subHistoId_cAxialDiff (2D numpy array, Integer)[-1, 2]: Detectors Id of the 
			projection that correspond with the current axial difference.
		"""
		eventsDetId_cAxialDiff = _eventsDetId[_eventsListModePos_cAxialDiff]
		
		for cAxial in range(self.nbDetAxial - _cAxialDiff):
			if self.vMode > 1:
				print("Current minimum axial Id: " + str(cAxial))
			eventInCurrBlock = np.where(\
						_eventsBlockId[_eventsListModePos_cAxialDiff][:, 1] == cAxial)[0]
							
			firstIdRel_cAxialDiff = _firstProjIdPerBlock[cAxial, _cAxialDiff] \
											- _firstProjId_cAxialDiff
			# Could use a better upper limit estimation but whatever
			lastIdRel = firstIdRel_cAxialDiff + _supProjInBlock

			histoId_cBlock = _subHistoId_cAxialDiff[firstIdRel_cAxialDiff:lastIdRel, :]

			for i, cEventDetId in enumerate(eventsDetId_cAxialDiff[eventInCurrBlock]):

				findProjIdOfEvents(_eventProjId, histoId_cBlock, cEventDetId, \
									_eventsListModePos_cAxialDiff[eventInCurrBlock][i], \
									_firstProjId_cAxialDiff + firstIdRel_cAxialDiff)


	def processCurrAxialDiffParallel(self, _eventProjId, _cAxialDiff, _eventsBlockId, \
									_eventsListModePos_cAxialDiff, _firstProjIdPerBlock,\
									_firstProjId_cAxialDiff, _supProjInBlock, \
									_eventsDetId, _subHistoId_cAxialDiff):
		"""
		Def.: Find the projection Id for the current difference in Axial Id. To make use 
		 	of MultiProcessing, some variables needs to be duplicated which will 
			increase the memory taken.
		@_eventProjId (MultiProcessing Array, Integer): The array that will contain the 
			projection Id of the events.
			(See processCurrAxialDiff())
		"""
		eventsDetId_cAxialDiff = _eventsDetId[_eventsListModePos_cAxialDiff]
		eventsBlockId_cAxialDiff = _eventsBlockId[_eventsListModePos_cAxialDiff]
		processArray = []
		for cAxial in range(self.nbDetAxial - _cAxialDiff):
			if self.vMode > 1:	
				print(cAxial)
				
			firstProjIdRel_cAxialDiff = _firstProjIdPerBlock[cAxial, _cAxialDiff] \
											- _firstProjId_cAxialDiff
			# Could use a better upper limit estimation.
			lastProjIdRel = firstProjIdRel_cAxialDiff + _supProjInBlock
			
			histoId_cBlock = \
					_subHistoId_cAxialDiff[firstProjIdRel_cAxialDiff:lastProjIdRel, :]	
			eventsRelPos_cBlock = np.where(eventsBlockId_cAxialDiff[:, 1] == cAxial)[0]	
			firstProjId_cBlock = _firstProjIdPerBlock[cAxial, _cAxialDiff]	
			eventsDetId_cBlock = eventsDetId_cAxialDiff[eventsRelPos_cBlock]	
			eventsPosInListMode = _eventsListModePos_cAxialDiff[eventsRelPos_cBlock]
				
			processArray.append(mp.Process(target=self.FindEventsProjIdCurrBlock, \
						args=(_eventProjId, eventsDetId_cBlock, eventsPosInListMode, \
							firstProjId_cBlock, histoId_cBlock)))
			processArray[cAxial].start()
			
			# Weak attempt to limit the number of process launched at a time.
			if (cAxial % self.nbProcToLaunch == 0) and (cAxial != 0):
				processArray[cAxial].join()
		# Make sure that every process of the current difference in axials Id are 
		# finished before continuing.
		for cAxial in range(self.nbDetAxial - _cAxialDiff):
			processArray[cAxial].join()
	
	
	def FindEventsProjIdCurrBlock(self, _eventProjId, _eventsDetId_cBlock, \
									_eventsPosInListMode, _firstProjId_cBlock, \
									_histoId_cBlock):
		""" 
		Def.: Find the projection Id of the events in the current block.
		Note: using "@jit(cache=True" make it slower, for some reason...
		@_eventProjId (1D MultiProcessing Array, Integer): Array that hold the 
			projection Id of the events.
		@_eventsDetId_cBlock (2D numpy array, Integer)[-1, 2]: Detectors Id of the  
			events in the current block.
		@_eventsPosInListMode (1D numpy array, Integer): Position in _eventProjId of the 
			events in the current block.
		@_firstProjId_cBlock (Integer): The projection Id of the first projection in 
			_histoId_cBlock.
		@_histoId_cBlock (2D numpy array, Integer)[-1, 2]: The detectors Id of the 
			projection for the current block.
		"""
		
		for i in range(_eventsDetId_cBlock.shape[0]):
			eventsDetId = _eventsDetId_cBlock[i]
			findProjIdOfEvents(_eventProjId, _histoId_cBlock, eventsDetId, \
								_eventsPosInListMode[i], _firstProjId_cBlock)


	def extractEventDetIdAndBlockId(self, _listModePath, _iMode, _timeSlice=None):
		"""
		Def.: Extract the event Id from the List Mode and their block Id.
		@_listModePath (String): Path to the file holding the data set in List Mode.
		@_iMode (String): Type of List Mode saved in _listModePath.
		@_timeSlice dsa
		Return
			eventsDetId (2D numpy array, Integer)[-1, 2]
			eventsBlockId (2D numpy array, Integer)[-1, 3]
		"""
		if self.vMode > 0:
			print("Loading events from listMode.")			
		
		# List mode obtained using savantDoiGateToListModeConverter with the option 
		# exTimeInfo.
		if _iMode == "basicWithoutTime":
			eventsDetId = np.fromfile(_listModePath, dtype='int32').reshape((-1, 2))
		# List mode obtained using savantDoiGateToListModeConverter default output.
		elif _iMode == "basic":
			eventsInfo = np.fromfile(_listModePath, dtype='float32').reshape((-1, 3))
			if _timeSlice != None:
				validEvent = np.where((eventsInfo[:, 0] > _timeSlice[0])
				                       & (eventsInfo[:, 0] < _timeSlice[1]) )[0]
				eventsDetId = eventsInfo[validEvent, 1:].astype('int32')
			else:
				eventsDetId = eventsInfo[:, 1:].astype('int32')
		# Castor list mode file.
		elif _iMode == "castor":
			eventCastorDetId = np.fromfile(_listModePath, \
												dtype='uint32').reshape((-1, 3))
			eventsDetId = eventCastorDetId[:, 1:].astype('int32')
			eventsDetId.sort(axis=1)
		else:
			print(str(_iMode) + " mode is not implemented yet")
			sys.exit(1)
			
		nbDetPerLayer = self.nbDetAxial * self.nbDetRadial

		eventsBlockId = np.empty((eventsDetId.shape[0], 3), dtype='int32')
		# Smallest axial Id of each coincident events.
		eventsBlockId[:, 1] = ((eventsDetId % nbDetPerLayer) \
									// self.nbDetRadial).min(axis=1)
		# Positive difference in axial Id of each coincident events.
		eventsBlockId[:, 2] = ((eventsDetId % nbDetPerLayer) \
								// self.nbDetRadial).max(axis=1) - eventsBlockId[:, 1]
		# Encoding of each event layer Id combinaison.
		# Note: Currently not used.
		eventsBlockId[:, 0] = (eventsDetId // nbDetPerLayer \
									* np.array([2, 1])).sum(axis=1)	
		# Events layer Id combinaison of 1 could either be [0, 1] or [1, 0]. We affected 
		# layer Id combinaison of 2 to the second cases.
		eventsBlockId[np.where((eventsBlockId[:, 0] == 1) \
				* (np.diff(eventsDetId % nbDetPerLayer, axis=1)[:,0] < 0))[0] , 0] += 1
		
		return eventsDetId, eventsBlockId


	def loadHistogramId(self, _histoIdPath, _cMode):
		"""
		Def.: Load the histogram projection Id. 
		@_histoIdPath (String): Path to the histogram Id.
		@_cMode (String): Computation mode to use. Either "ram" or "onDisk". Currently,
		 	it is one or the other for all the variable.
		"""
		
		if _histoIdPath != "":
			if self.vMode > 0:
				print("Loading histogram detector Id")
			
			dataType = np.dtype([('d1',np.int32), ('d2',np.int32)])
			nbProj = int(os.path.getsize(_histoIdPath) / (dataType.itemsize))
			if _cMode == "onDisk":
				self.histogramId = np.memmap(_histoIdPath, dtype='int32', mode='r', \
											shape=(nbProj, 2))
			elif _cMode == "ram":
				self.histogramId = \
						np.fromfile(_histoIdPath, dtype='int32').reshape((nbProj, 2))


	def initHistogram(self, _histoPath, _cMode):
		"""
		Def.: Initialize the histogram to zero.
		@_histoPath (String): Path where the histogram will be saved.
		@_cMode (String): Computation mode to use. Either "ram" or "onDisk". Currently,
		 	it is one or the other for all the variable.
		"""
		if self.vMode > 0:
			print("Initializing the histogram")
		
		if _cMode == "onDisk":
			self.histogram = np.memmap(_histoPath, dtype='float32', mode='w+', \
										shape=(self.histogramId.shape[0]))
			self.histogram[...] = 0.0	
		elif _cMode == "ram":
			self.histogram = np.zeros(self.histogramId.shape[0], dtype='float32')


	def genBasisProjIdDict(self, _firstProjIdPerBlock):
		"""
		Def.: Create the dictionnary that will hold the relative projection Id of the 
			valid projections that have both detector on the first axial element and 
			those that have one detector on the first axial element and the other on the 
			second axial element.
			Since the sameAxialId is the first block of the projections Id, it is also 
			its absolute projection Id.
		@_firstProjIdPerBlock (2D numpy array, Integer)[-1, 3]: The layer Id 
			combinaison, minimum axial Id and difference in axial Id of all the events.
		Return: Dict (Integer, Integer)
		"""
		if self.vMode > 0:
			print("Creating the basis projection Id dictionnary.")
		# The first case of difference in axial Id is those with minimum Axial Id 0 and
		# Axial difference Id of 0.
		sameAxialBasis = self.histogramId[0:_firstProjIdPerBlock[1, 0], :]
		# The first case of difference in axial Id is those with minimum Axial Id 0 and
		# Axial difference Id of 1.
		diffAxialBasis = \
				self.histogramId[_firstProjIdPerBlock[0,1]:_firstProjIdPerBlock[1,1], :]
		
		# projIdBasisDict = {}
		projIdBasisDict = Dict.empty(key_type=types.int64, value_type=types.int64)
		nbDetTot = self.nbDetAxial * self.nbDetLayer * self.nbDetRadial
		
		# Fill the sameAxialBasis case.
		currProj = 0
		for detId in sameAxialBasis:
			projIdBasisDict[detId[0] + nbDetTot * detId[1]] = currProj
			currProj += 1
		
		# Fill the diffAxialBasis case.
		currProj = 0
		for detId in diffAxialBasis:
			projIdBasisDict[detId[0] + nbDetTot * detId[1]] = currProj
			currProj += 1	
		
		return projIdBasisDict

	def genProjBlockPartition(self):
		"""
		Def.: Generate the first projection Id of each block and a supremum of the 
			number of projection in any block. 
		Return: 
			firstProjIdPerBlock (2D numpy array)[nbDetAxial, maxAxialDiff + 1]
			supProjInBlock (Integer)
		"""
		# Since we didn't write the number of projection per radial basis, we 
		# re-compute it.
		radialBasisProjDetId_sameRing, radialBasisProjDetId_diffRing \
															= self.genRadProjDetIdBasis()
		totalNbElementDiffLayer = radialBasisProjDetId_diffRing.shape[0] 
		totalNbElementSameLayer = radialBasisProjDetId_sameRing.shape[0]
		
		nbProjPerRingSameRing = totalNbElementDiffLayer + 2 * totalNbElementSameLayer
		nbProjPerRingDiffRing = 4 * totalNbElementDiffLayer
		
		nbProjPerAxialDiff = np.zeros(self.maxAxialDiff + 1, dtype='int64')
		# The number of unique projection is different when the axial Id difference is
		# zero.
		nbProjPerAxialDiff[1] = nbProjPerRingSameRing * self.nbDetAxial
		nbProjPerAxialDiff[2:] = nbProjPerRingDiffRing \
									* (self.nbDetAxial - np.arange(1, self.maxAxialDiff))
									
		firstProjIdPerBlock = np.zeros((self.nbDetAxial, self.maxAxialDiff + 1), \
									dtype='int64')
		firstProjIdPerBlock += np.cumsum(nbProjPerAxialDiff)[np.newaxis, :]
		# The number of unique projection is different when the axial Id difference is
		# zero.
		firstProjIdPerBlock[:, 0] += np.arange(0, self.nbDetAxial)[:] \
																* nbProjPerRingSameRing
		firstProjIdPerBlock[:, 1:] += np.arange(0, self.nbDetAxial)[:, np.newaxis] \
																* nbProjPerRingDiffRing		
		# Too complex to correctly have the last projection so we take a value we know 
		# is equal or larger than the number of projection in any block. 
		# This does not change anything as long as the event is in the block. If not,
		# it will search longer for nothing.
		supProjInBlock = nbProjPerRingDiffRing
			
		return firstProjIdPerBlock, supProjInBlock


	def writeEventsInHistogram(self, _eventProjId, _histoPath, _cMode):
		""" 
		Def.: Add the events to the histogram. 
		@_eventProjId (1D numpy array, Integer): The projection Id of the event 
			extracted from the list mode. All valid event are stord one after another so
			the first -1 found indicate that all valid event where already taken care of.
		@_histoPath (String): Path where the histogram will be saved.
		@_cMode (String): Computation mode used. If the histogram was kept in ram, we 
			need to write it to the file. 
		"""
		if self.vMode > 0:
			print("Writing events in the histogram.")		
			
		for event in _eventProjId:
			if event != -1:
				self.histogram[event] += 1 
			
		if _cMode == "ram":
			self.histogram.tofile(_histoPath)


#########################################################################################
# Function of this module.

@jit(cache=True)
def findProjIdOfEvents(_eventProjId, _currBlockOfProjDetId, _eventDetId, \
							_eventPosInListMode, _projIdOffSetOfCurrBlock):
	""" 
	Def.: Search for the position, relative to _currBlockOfProjDetId, to which the 
		current event detector Id correspond to.
	@_eventDetId (1D numpy array, Integer): The detector Id of the current event to
		process.
	@_currBlockOfProjDetId (2D numpy array, Integer): A block of projection detector Id.
	@_eventProjId (1D numpy array, Integer): The projection Id of the events.
	@_eventPosInListMode (Integer): The current number of valid event processed.
	@_projIdOffSetOfCurrBlock (Integer): The projection Id of first projection in 
		_currBlockOfProjDetId.
	"""
	nbProjCurrBlock = _currBlockOfProjDetId.shape[0]
	
	relProjPos = -1
	for j in range(nbProjCurrBlock):
		if (_currBlockOfProjDetId[j, 0] == _eventDetId[0]) \
				and (_currBlockOfProjDetId[j, 1] == _eventDetId[1]):
			relProjPos = j
			break
			
	if relProjPos != -1:
		_eventProjId[_eventPosInListMode] = _projIdOffSetOfCurrBlock + relProjPos
			


@njit(cache=True)
def findRelProjIdOfEventList(_eventProjId, _currBlockOfProjDetId, _eventList, \
								_nbValidEvent, _projIdOffSetOfCurrBlock):
	""" 
	Def.: Search for the positions, relative to _currBlockOfProjDetId, to which the 
		current events detector Id correspond to. 
	@_eventList (2D numpy array, Integer)[-1, 2]: The detector Id of the events to
		process.
	@_currBlockOfProjDetId (2D numpy array, Integer): A block of projection detector Id.
	@_eventProjId (1D numpy array, Integer): The projection Id of the events.
	@_nbValidEvent (Integer): The current number of valid event processed.
	@_projIdOffSetOfCurrBlock (Integer): The projection Id of first projection in 
		_currBlockOfProjDetId.
	Return:
		_eventProjId (1D numpy array, Integer)
		_nbValidEvent (Integer)
	"""
	
	nbProjCurrBlock = _currBlockOfProjDetId.shape[0]
	nbEvent = _eventList.shape[0]
	
	for i in range(nbEvent):
		relProjPos = -1
		for j in range(nbProjCurrBlock):
			if (_currBlockOfProjDetId[j, 0] == _eventList[i, 0]) \
					and (_currBlockOfProjDetId[j, 1] == _eventList[i, 1]):
				relProjPos = j
				break
				
		if relProjPos != -1:
			_eventProjId[_nbValidEvent] = _projIdOffSetOfCurrBlock + relProjPos
			_nbValidEvent += 1

	return _eventProjId, _nbValidEvent


@njit
def findProjIdOfEventsDict(_eventProjId, _projIdBasisDict, _eventsBlockId, \
                           _mergedDetId, _firstProjIdPerBlock, _maxAxialDiff, \
                           _verboseLevel):
	"""
	Def.: Find the projection Id of the events using a precomputed dictionnary.
	@_eventProjId (1D numpy array, Integer): Where the projection Id of the events will 
		be saved.
	@_projIdBasisDict (Dict (Integer, Integer)): The projection Id of the detector pairs 
		which are in the radial basis. 
	@_eventsBlockId (2D numpy array, Integer)[-1, 3]: The block Id of each events.
	@_mergedDetId (1D numpy array, Integer): The merged Id of the detector pair of each 
		events.
	@_firstProjIdPerBlock (2D numpy array, Integer)[nbDetAxail, maxAxialDiff-1]: The 
		projection Id of the first projection of each block of axial Id and difference 
		in axial Id. 
	@_maxAxialDiff (Integer): The maximum axial difference accepted.
	@_verboseLevel (Integer): Level of verbosity.
	"""
	for i in range(_eventProjId.size):
		if (_verboseLevel > 1) and ((i % (_eventProjId.size // 1000)) == 0):
			print(100.0 * float(i) / float(_eventProjId.size))
		# Check if the current _mergedDetId is in the dictionnary created. If not, it
		# means that it was excluded using the radial FoV and should be ignored.
		# Also check for events that exceed the maximum axial difference.
		if (_mergedDetId[i] in _projIdBasisDict) \
				and (_eventsBlockId[i, 2] < (_maxAxialDiff + 1)):
			cEventBasisProjId = _projIdBasisDict[_mergedDetId[i]]
			_eventProjId[i] = cEventBasisProjId \
					+ _firstProjIdPerBlock[_eventsBlockId[i, 1], _eventsBlockId[i, 2]]


def readArguments():
	'''
	Def : Parse command line arguments.
	Return : Provided arguments.
	'''
	parser = argparse.ArgumentParser(description='Convert a list mode into a '
						'histogram assuming the default configuration of the Savant.')
	parser.add_argument('-iListMode', dest='listModeFile', type=str, required=True, \
						help='Path to the list mode to convert.')
	parser.add_argument('-iHistoId', dest='histoIdFile', type=str, required=True, \
						help='Path to the histogram Id.')
	parser.add_argument('-oHist', dest='histogramFile', type=str, required=True, \
						help='Where the histogram will be saved.')
	parser.add_argument('-cMode', dest='cMode', type=str, required=True,\
						choices=["ram", "onDisk"], \
						help='Specify if the script should work in ram or on disk. '\
							'Faster on ram, but it can take much space')
	parser.add_argument('-iMode', dest='iMode', type=str, required=False,\
						choices=["basic", "basicWithoutTime", "castor"],  \
						default="basic", \
						help='The type of encoding used of the list Mode.')
	parser.add_argument('-timeSlice', dest='timeSlice', type=float, required=False,\
						nargs=2, default=None, \
						help='Give the time slice from which we want the data to be ' \
						     'extracted.')
	# parser.add_argument('-nbProc', dest='nbProc', type=int, required=False, default=1, \
	# 					help='The number of process to use.')
	parser.add_argument('-v', dest='verbosity', type=int, required=False, default=0, \
						help='The level of verbosity.')
	
	return parser.parse_args()



#########################################################################################
# Main : We use this to make the main usable has a script and a method.
#########################################################################################
if __name__== "__main__":
	"""
	Currently, this module script functionality is only to convert a listMode to the 
	default Savant histogram. The module can handle other cases, not the script.
	"""
	args = readArguments()
	
	histoBuilder = HistogramBuilder()
	histoBuilder.setHistogramConfigToSavantDefault()
	histoBuilder.vMode = args.verbosity
	
	histoBuilder.convertListModeToHistogramDict(args.listModeFile, args.histogramFile, \
	                                            args.histoIdFile, _cMode=args.cMode, \
	                                            _iMode=args.iMode, \
												_timeSlice=args.timeSlice, \
												_nbProcess=1)

	