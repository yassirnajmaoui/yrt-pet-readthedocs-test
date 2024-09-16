#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module used to convert a Gate created simulation dataset into a format that can be read 
by the histogramBuilder Class. Currently, this is a compilation of methods which are 
called from the main. Generalizing this module to a Class is useless since it is 
specific for the application.

TODO:
	- Add arrival time (histogramBuilder need to be modified for that)
"""

#########################################################################################
# Imports
import numpy as _np
import sys as _sys
import os as _os
import argparse as _argparse

import gateReader as gR



#########################################################################################
# List of methods created to build the features.

def convertGateDetIdToBasisDetId(_eventGateDetId):
	''' 
	Def.: Convert the detector Id of coincident events obtained from a Gate simulation 
		of the Savant DOI into (Layer, Axial, Radial) Id.
	Note: Savant DOI: GantryId: 1; RSector: [56, 1]; Module: [1, 18];
						SubModule: [2, 2]; Detector: [8, 4]; Layer: [2]
	@_eventGateDetId (3D numpy array, Int32)[-1, 2, 6]: Array of detector Id from 
		coincident events obtained from a Gate simulation.
	Return: (2D numpy array, Int32)[-1, 2, 3]
	'''
	eventBasicDetId = _np.empty([_eventGateDetId.shape[0], 2, 3], dtype='int32')

	# Layer dimension.
	# The 6th dimension is directly the layer Id.
	eventBasicDetId[:, :, 0] = _eventGateDetId[:, :, 5]
	
	# Axial dimension.
	axialModuleId = _eventGateDetId[:, :, 2] 
	nbCystalAxialInSubModule = 4
	nbCystalAxialInModule = nbCystalAxialInSubModule * 2
	subModuleRowInModule = _eventGateDetId[:, :, 3] // 2
	cystalRowInSubModule = _eventGateDetId[:, :, 4] // 8
	eventBasicDetId[:, :, 1] = axialModuleId * nbCystalAxialInModule \
									+ subModuleRowInModule * nbCystalAxialInSubModule \
									+ cystalRowInSubModule
									
	# Radial dimension.
	radialModuleId =  _eventGateDetId[:, :, 1] 
	nbCystalRadialInSubModule = 8
	nbCystalRadialInModule = nbCystalRadialInSubModule * 2
	subModuleColInModule = _eventGateDetId[:, :, 3] % 2
	cystalColInSubModule = _eventGateDetId[:, :, 4] % 8
	eventBasicDetId[:, :, 2] = radialModuleId * nbCystalRadialInModule \
									+ subModuleColInModule * nbCystalRadialInSubModule \
									+ cystalColInSubModule
		
	return eventBasicDetId


def convertGateDetIdToCastorDetId(_eventGateDetId, _sortProjDetId=True, \
									_eventPreprocessor=None):
	'''
	Def.: Convert the detector Id of coincident events obtained from a Gate simulation 
		of the Savant DOI into castor-like detector Id. If _eventPreprocessor is 
		defined, it is used to remove events following its rule.
	Note: Gate unwrap the detector Id from the smallest elements to the highest one.
	@_eventGateDetId (3D numpy array, Int32)[-1, 2, 6]: Array of detector Id from 
		coincident events obtained from a Gate simulation.
	@_sortProjDetId (Boolean): Indicates if the detector Id of each projections should be
		sorted.
	@_eventPreprocessor (Function): A method which remove some coincident events  
		following certain rules.
	Return: 
		(2D numpy array, Int32)[-1, 2]
		(1D numpy array, Int32)[-1]
	'''
	
	eventBasicDetId = convertGateDetIdToBasisDetId(_eventGateDetId)
	
	if _eventPreprocessor != None:
		eventBasicDetId, validEvent = _eventPreprocessor["func"](eventBasicDetId, \
														_eventPreprocessor["opts"])
	
	# Detector Id creation.
	# The un-wrapping is Radial -> Axial -> Layer. (Layer change last.)
	# Dimensions: DOI, Axial, Radial
	nbDetAxial = 144
	nbDetRadial = 896
	eventCastorDetId = eventBasicDetId[:, :, 0] * nbDetAxial * nbDetRadial \
							+ eventBasicDetId[:, :, 1] * nbDetRadial \
							+ eventBasicDetId[:, :, 2]
							
	if _sortProjDetId == True:
		eventCastorDetId.sort(axis=1)
		
	return eventCastorDetId, validEvent


def approximateEventEmissiontime(_eventCoincArrTime, _validEvent):
	'''
	Def.: Approximate the emission time of the events that are considered valid.
	@_eventCoincArrTime (2D numpy array, Float)[-1, 2]: Arrival times of the coincident 
		events obtained from a Gate simulation.
	@_validEvent (1D numpy array, Integer): Index of the events that were kept when the 
		detector Id of the events were extracted.
	Return: 
		(2D numpy array, Int32)[-1, 2]
	'''
	# We take the smallest arrival time to approximate the events emission time. 
	# The difference between the arrival time of the events should be in the order of 
	# the nanoseconds, so it does not really matter to take the minimum or one of them.
	eventArrTimes = _np.min(_eventCoincArrTime[_validEvent], axis=1)
	
	return eventArrTimes



def removeProjWithToLargeRingDiff(_eventBasicDetId, _maxRingDiff):
	'''
	Def.: Create a copy of _eventBasicDetId where coincident events with a axial Id 
		difference higher than _maxRingDiff were removed.
	@_eventGateDetId (3D numpy array, Int32)[-1, 2, 6]: Array of detector Id from 
		coincident events obtained from a Gate simulation.
	@_maxRingDiff (Integer) : The maximum ring difference accepted for a coincidence.
	Return: 
		(2D numpy array, Int32)[-1, 2, 3]
		(1D numpy array, Int32)[-1]
	'''
	nbDetAxial = 144

	coincAxialIdMin = _eventBasicDetId[:, :, 1].min(axis=1)
	coincRingDiff = _eventBasicDetId[:, :, 1].max(axis=1) - coincAxialIdMin

	insideFOV = _np.where( (coincRingDiff <= _maxRingDiff) \
							* (coincAxialIdMin + coincRingDiff <= nbDetAxial))
							
	return _eventBasicDetId[insideFOV], insideFOV


def convertGateBinaryCoincFileToListMode(_gateBinaryFile, _maxNbLinesLoadedAtOneTime, \
											_dataType, _eventPreprocessor, \
											_exTimeInfo=False):
	'''
	Def.: Convert the coincidence in a Gate Binary coincidence file into list mode.
	@_gateBinaryFile (String): Path to the binary coincidence file holding the events.
	@_maxNbLinesLoadedAtOneTime (Integer): Set the maximum number of coincidence events 
		loaded at one time.
	@_dataType (numpy.dtype): Defines the order and type of each line saved in the Gate 
		file.
	@_eventPreprocessor (Function): A method which remove some coincident events  
		following certain rules.
	@_exTimeInfo (Boolean): Flag that indicate if the the arrival time information 
		should not be kept.
	Return: (2D numpy array, Int32)[-1, 2]
			or (2D numpy array, Float)[-1, 3]
	'''
	# Set the maximum size of the parts that will be processed.
	nbPartToProcess = int(_np.ceil(_os.path.getsize(_gateBinaryFile) \
									/ (_maxNbLinesLoadedAtOneTime * _dataType.itemsize)))	
	
	listOfPartConverted_detId = []
	if _exTimeInfo == False:
		listOfPartConverted_arrTime = []
	for iPart in range(nbPartToProcess):
		# Extract a part of the Gate coincidences file.
		dataGlob = gR.extractDataFromGateBinaryFile(_gateBinaryFile, _dataType,
													_maxNbLinesLoadedAtOneTime, iPart)
		# Convert the current part.
		eventCastorDetId, validEvent = convertGateDetIdToCastorDetId(\
												dataGlob["gateDetId"], \
												_eventPreprocessor=_eventPreprocessor)
		listOfPartConverted_detId += [eventCastorDetId]
		if _exTimeInfo == False:
			listOfPartConverted_arrTime += [approximateEventEmissiontime(\
															dataGlob["arrTime"], \
															validEvent)[:, _np.newaxis]]
		
	eventCastorDetId = _np.vstack(listOfPartConverted_detId)
	if _exTimeInfo == False:
		eventArrTimes = _np.vstack(listOfPartConverted_arrTime).astype('float32')
		eventListMode = _np.hstack((eventArrTimes, eventCastorDetId)).astype('float32')
	else: 
		eventListMode = eventCastorDetId
	
	return eventListMode
	



#########################################################################################
# List of methods used to use this module as a script.


def readArguments():
	'''
	Def : Parse command line arguments.
	Return : Provided arguments.
	'''
	parser = _argparse.ArgumentParser(description='Convert Gate coincidence data set ' \
					'obtained from simulation of the Savant DOI in a histogram ' \
					'format. If multiple files are given, convert and merged all of ' \
					'them.')
	parser.add_argument('-iFile', dest='inputFile', type=str, required=True, nargs='+', \
						help='Path to the Gate file.')
	parser.add_argument('-oFile', dest='outputFile', type=str, required=True, \
						help='Path to the file where the converted detector Id will ' \
							'be saved.')
	parser.add_argument('-dataType', dest='dataType', type=str, \
						required=True, choices=["basic", "basic_Gt", "basic_Gt_Inter"], \
						help='The data type used to save the coincidence in Gates.')
	parser.add_argument('-maxNbLineread', dest='maxNbLineread', type=int, \
						required=False, default=1000000, \
						help='The maximum number of lines that should be loaded from ' \
							'the Gate binary files.')
	parser.add_argument('-rmCoincRingDiff', dest='rmCoincRingDiff', type=int, \
						required=False, default=-1, \
						help='Flag to remove all events that have a higher ring ' \
							'difference than the user specification.')
	parser.add_argument('-excludeTimeInformation', dest='exTimeInfo', \
						action='store_true', required=False, default=False, \
						help='Flag to not keep the time information of the events.')
	
	return parser.parse_args()




#########################################################################################
# Main : We use this to make the main usable has a script and a method.
#########################################################################################

if __name__== "__main__":
	"""
	Currently, all of this module feature are accessible in script mode.
	"""

	args = readArguments()
	
	if args.dataType == "basic":
		dataType = gR.GateBinaryDataType
	elif args.dataType == "basic_Gt":
 		dataType = gR.GateBinaryDataTypeWithSourcePos
	elif  args.dataType == "basic_Gt_Inter":
		dataType = gR.GateBinaryDataTypeWithSourcePosAndGammaInter

	if args.rmCoincRingDiff != -1:
		cleanEvent = {'func': removeProjWithToLargeRingDiff, \
						'opts': args.rmCoincRingDiff}
	else:
		cleanEvent = None

	listOfConvertedFile = []
	for cFile in args.inputFile:
		listOfConvertedFile += [convertGateBinaryCoincFileToListMode(cFile, \
																args.maxNbLineread,
																dataType, cleanEvent,
																args.exTimeInfo)]
	eventCastorDetId = _np.vstack(listOfConvertedFile)
	
	eventCastorDetId.tofile(args.outputFile)
															
