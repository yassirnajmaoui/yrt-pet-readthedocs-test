#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Goal of this module:
	Tools to extract data from a Gate simulated acquisition.

	The position of the variables of interest were found in the section "Data output" of
	the Gate Users Guide version 7.1.
'''




#########################################################################################
# List of import needed
#########################################################################################
# Container used for the extracted data.
import numpy as _np
# Tools to access the raw data for extraction.
import pandas as _pd
# Used to exit code when an error occur.
import sys as _sys
# Used to access the size, in bytes, of a file to infer its number of variable.
import os as _os
# Just for a good approximation of the speed of light.
from scipy import constants as _constants
import copy

import argparse


#########################################################################################
# Constant of this module
#########################################################################################
# Data type used for each variable kept for each event. It is needed to decode a Gate
# simulation saved in the binary format.
GateBinaryDataType = _np.dtype( [('t0',_np.float64), ('Camera0',_np.int32), \
					('Ring0',_np.int32), ('Bloc0',_np.int32), ('Mod0',_np.int32), \
					('Det0',_np.int32), ('subDet0',_np.int32), \
					('t1',_np.float64), ('Camera1',_np.int32), ('Ring1',_np.int32), \
					('Bloc1',_np.int32), ('Mod1',_np.int32), ('Det1',_np.int32), \
					('subDet1',_np.int32)] )

GateBinaryDataTypeWithSourcePos = _np.dtype( [('X0',_np.float64), ('Y0',_np.float64), \
					('Z0',_np.float64), ('t0',_np.float64), ('Camera0',_np.int32), \
					('Ring0',_np.int32), ('Bloc0',_np.int32), ('Mod0',_np.int32), \
					('Det0',_np.int32), ('subDet0',_np.int32), \
					('X1',_np.float64), ('Y1',_np.float64), ('Z1',_np.float64), \
					('t1',_np.float64), ('Camera1',_np.int32), ('Ring1',_np.int32), \
					('Bloc1',_np.int32), ('Mod1',_np.int32), ('Det1',_np.int32), \
					('subDet1',_np.int32)] )

GateBinaryDataTypeWithSourcePosAndGammaInter = \
		_np.dtype( [('X0',_np.float64), ('Y0',_np.float64), ('Z0',_np.float64), \
					('t0',_np.float64), \
					('P0_X',_np.float64), ('P0_Y',_np.float64), ('P0_Z',_np.float64), \
					('Camera0',_np.int32), ('Ring0',_np.int32), ('Bloc0',_np.int32), \
					('Mod0',_np.int32), ('Det0',_np.int32), ('subDet0',_np.int32), \
					('X1',_np.float64), ('Y1',_np.float64), ('Z1',_np.float64), \
					('t1',_np.float64), \
					('P1_X',_np.float64), ('P1_Y',_np.float64), ('P1_Z',_np.float64), \
					('Camera1',_np.int32), ('Ring1',_np.int32), ('Bloc1',_np.int32), \
					('Mod1',_np.int32), ('Det1',_np.int32), ('subDet1',_np.int32)] )

dictOfGateBinaryDataType = { "GateBinaryDataType": GateBinaryDataType, \
				"GateBinaryDataTypeWithSourcePos": GateBinaryDataTypeWithSourcePos,\
				"GateBinaryDataTypeWithSourcePosAndGammaInter": \
					GateBinaryDataTypeWithSourcePosAndGammaInter}


#########################################################################################
# List of the methods implemented in this module.
#########################################################################################

def extractSimulDataFromGateTextFile(_pathToFile, _groundTruth=False):
	'''
	Def: Extract the arrival times and detector Id of event simulated from Gate and saved
	     in text format.
	@_pathToFile (String): Path to the file where the results of a Gate simulation were
		saved.
	Return:
		photonPairArrTime[nbCoinc, 2] (2D numpy array, Float)
		photonPairGateDetId[nbCoinc, 2, posGateSpace==2] (3D numpy array, Integer)
	Warning: posGateSpace is equal to two in the 2D acquisition setup. Most likely, this
		will change when the setup will be in 3D.
	'''
	# Load the Gate data saved un a text file.
	gateData = _np.loadtxt(_pathToFile)

	# Initialize both array.
	photonPairArrTime = _np.empty([gateData.shape[0], 2])
	photonPairGateDetId = _np.empty([gateData.shape[0], 2, 6])

	# Infer the position of the variables from the total number of variable available.
	if gateData.shape[1] == 14:
		# Only the arrival time and detector Id were kept.
		indexArrTime = (0, 7)
		indexDetId_t0 = (2, 3)
		indexDetId_t1 = (9, 10)
	elif gateData.shape[1] == 20:
		# Only the arrival time, detector Id  and source origin were kept.
		indexArrTime = (3, 13)
		indexDetId_t0 = (5, 6)
		indexDetId_t1 = (15, 16)
		indexSourcePos = (0, 1, 2)
	elif gateData.shape[1] == 46:
		# All variable were kept.
		indexArrTime = (6, 29)
		indexDetId_t0 = (11, 12, 13, 14, 15, 16)
		indexDetId_t1 = (34, 35, 36, 37, 38, 39)
		indexSourcePos = (3, 4, 5)
	else:
		print("The number of variable in the file " + str(_pathToFile) + " does not fit " +
				"any of the known configuration. Exiting")
		_sys.exit(1)

	# Extract the values of interest.
	photonPairArrTime = gateData[:, indexArrTime]
	photonPairGateDetId[:, 0, :] = gateData[:, indexDetId_t0]
	photonPairGateDetId[:, 1, :] = gateData[:, indexDetId_t1]

	if _groundTruth == True:
		photonSourcePosition = _np.empty([gateData.shape[0], 3])
		photonSourcePosition = gateData[:, indexSourcePos]
		return photonPairArrTime, photonPairGateDetId, photonSourcePosition
	else:
		return photonPairArrTime, photonPairGateDetId




def extractDataFromGateBinaryFile(_pathToFile, _dataType, _nbDataInChunk=-1, \
									_chunckId=-1):
	'''
	Def: Extract the information of events simulated from Gate and saved at _pathToFile 
		in binary format. _dataType gives the format from which the binary can be 
		interpreted.
	@_pathToFile (String): Path to the file where the results of a Gate simulation were
		saved.
	@_dataType (Element of dictOfGateBinaryDataType): A numpy dataType that defines how 
		much bytes each information of an event takes.
	@_nbDataInChunk (Integer): The number of coincidences in each chunk.
	@_chunckId (Integer): Which chunk we want to extract.
	Return:
		Dictionnary holding all the data extracted from the Gate file.
	'''
	# Each events are saved in the same format, thus the number of bytes should be a
	# multiple of the number of bytes needed for one event.
	nbBytesInFile = _os.path.getsize(_pathToFile)
	if (nbBytesInFile % _dataType.itemsize != 0):
		print("The number of bytes is not a multiple of the number of bytes needed " +
				"for one event. The format used might not be the one known by this " +
				"script, so we terminate.")
		_sys.exit(1)

	# Extract binary values from the file.
	fileIO = open(_pathToFile, "rb")
	if _nbDataInChunk != -1:
		fileIO.seek(_chunckId * _nbDataInChunk * _dataType.itemsize, _os.SEEK_SET)

	gateData = _np.fromfile(fileIO, dtype=_dataType, count=_nbDataInChunk)
	# Convert previous variable to a column named matrix.
	gateData = _pd.DataFrame(gateData.tolist(), columns=_dataType.names)

	# Initialize the dictionnary that will hold the data.
	extGateData = {}
	
	if 't0' in _dataType.names:
		# Initialize both array.
		extGateData["arrTime"] = _np.empty([gateData.shape[0], 2])
		# Extract the values of interest.
		extGateData["arrTime"][:, 0] = gateData.ix[:, 't0'].as_matrix()
		extGateData["arrTime"][:, 1] = gateData.ix[:, 't1'].as_matrix()
	
	# All the detector Id elements are always there in my experience, so any of them 
	# works.
	if 'Det0' in _dataType.names:
		# Initialize both array.
		extGateData["gateDetId"] = _np.empty([gateData.shape[0], 2, 6])
		# Extract the values of interest.
		extGateData["gateDetId"][:, 0, 0] = gateData.ix[:, 'Camera0'].as_matrix()
		extGateData["gateDetId"][:, 0, 1] = gateData.ix[:, 'Ring0'].as_matrix()
		extGateData["gateDetId"][:, 0, 2] = gateData.ix[:, 'Bloc0'].as_matrix()
		extGateData["gateDetId"][:, 0, 3] = gateData.ix[:, 'Mod0'].as_matrix()
		extGateData["gateDetId"][:, 0, 4] = gateData.ix[:, 'Det0'].as_matrix()
		extGateData["gateDetId"][:, 0, 5] = gateData.ix[:, 'subDet0'].as_matrix()
		extGateData["gateDetId"][:, 1, 0] = gateData.ix[:, 'Camera1'].as_matrix()
		extGateData["gateDetId"][:, 1, 1] = gateData.ix[:, 'Ring1'].as_matrix()
		extGateData["gateDetId"][:, 1, 2] = gateData.ix[:, 'Bloc1'].as_matrix()
		extGateData["gateDetId"][:, 1, 3] = gateData.ix[:, 'Mod1'].as_matrix()
		extGateData["gateDetId"][:, 1, 4] = gateData.ix[:, 'Det1'].as_matrix()
		extGateData["gateDetId"][:, 1, 5] = gateData.ix[:, 'subDet1'].as_matrix()
		
	if 'X0' in _dataType.names:
		# Initialize both array.
		extGateData["sourcePos"] = _np.empty([gateData.shape[0], 2, 3])
		# Extract the values of interest.
		extGateData["sourcePos"][:, 0, 0] = gateData.ix[:, 'X0'].as_matrix()
		extGateData["sourcePos"][:, 0, 1] = gateData.ix[:, 'Y0'].as_matrix()
		extGateData["sourcePos"][:, 0, 2] = gateData.ix[:, 'Z0'].as_matrix()
		extGateData["sourcePos"][:, 1, 0] = gateData.ix[:, 'X1'].as_matrix()
		extGateData["sourcePos"][:, 1, 1] = gateData.ix[:, 'Y1'].as_matrix()
		extGateData["sourcePos"][:, 1, 2] = gateData.ix[:, 'Z1'].as_matrix()
	
	if 'P0_X' in _dataType.names:
		# Initialize both array.
		extGateData["interPos"] = _np.empty([gateData.shape[0], 2, 3])
		# Extract the values of interest.
		extGateData["interPos"][:, 0, 0] = gateData.ix[:, 'P0_X'].as_matrix()
		extGateData["interPos"][:, 0, 1] = gateData.ix[:, 'P0_Y'].as_matrix()
		extGateData["interPos"][:, 0, 2] = gateData.ix[:, 'P0_Z'].as_matrix()
		extGateData["interPos"][:, 1, 0] = gateData.ix[:, 'P1_X'].as_matrix()
		extGateData["interPos"][:, 1, 1] = gateData.ix[:, 'P1_Y'].as_matrix()
		extGateData["interPos"][:, 1, 2] = gateData.ix[:, 'P1_Z'].as_matrix()
	
	return extGateData



def extractGroundTruth(_pathToFile, _fileDataType, _nbVoxel, _imSize, \
                       _baseNameOfOutput, _timeBins=None, _cMode="ram"):
	"""
	Def.: Extract the groundtruth from a binary file created by a Gate simulation and 
		save it as an image. If time bins are defined, create a ground truth for the 
		each time slice defined.
	@_pathToFile (String): Path to the Gate binary file.
	@_fileDataType (String): The dataset description key that describe what is saved in 
		the Gate data file.
	@_nbVoxel (List of 3 float): The number of voxel in each dimension.
	@_imSize (List of 3 float): The size of the image in each dimension.
	@_baseNameOfOutput (String): Where the grountruth is saved. If _timeBins is defined,
		it is used as the base name for all the groundtruths. 
	@_timeBins (List Float): The time bins to use to divide the data.
	@_cMode (String): The computation mode used. "part" is used to load only one part of  
		the data at a time. Only usefull if ram is limited. The "ram" option load all 
		the data in ram.
	"""

	if _cMode == "part":
		maxNbLinesLoaded = 2000000
		nbChunck = int(_np.ceil(_os.path.getsize(_pathToFile) / (maxNbLinesLoaded \
						* dictOfGateBinaryDataType[_fileDataType].itemsize)))
	elif _cMode == "ram":
		maxNbLinesLoaded = -1
		nbChunck = 1
	else:
		_sys.exit("Computation mode " + str(cMode) + " is not valid.")
		
	coincOrigPart = []
	coincArrTimePart = []
	for cChunck in range(nbChunck):
		dataGlob = extractDataFromGateBinaryFile(_pathToFile, \
		             dictOfGateBinaryDataType[_fileDataType], maxNbLinesLoaded, cChunck)
		data = dataGlob['sourcePos']
		promptCoincIndex = _np.where(_np.all(data[:, 0, :] == data[:, 1, :], \
										axis=1))[0]
		coincOrigPart += [data[promptCoincIndex, 0, :3]]
		coincArrTimePart += [dataGlob['arrTime'][promptCoincIndex].mean(axis=1)[:, \
																		_np.newaxis]]
	
	coincOrig = _np.vstack(coincOrigPart)
	coincArrTime = _np.vstack(coincArrTimePart)
	
	imSize = [[-_imSize[0]/2.0, _imSize[0]/2.0], [-_imSize[1]/2.0, _imSize[1]/2.0], \
	          [-_imSize[2]/2.0, _imSize[2]/2.0]]
	
	if _timeBins != None:
		groundTruth_motionStateDivide = [None] * (len(_timeBins) - 1)
		timeBins = _timeBins
	else:
		groundTruth_motionStateDivide = [None]
		timeBins = [coincArrTime.min(), coincArrTime.max()]
	for m in range(len(timeBins) - 1):
		eventInCurrentMotionState = _np.where( (timeBins[m] <= coincArrTime) \
												* (coincArrTime < timeBins[m+1]))[0]
		groundTruth_motionStateDivide[m], _ = _np.histogramdd(\
												coincOrig[eventInCurrentMotionState], \
												bins=_nbVoxel, range=imSize)
		
	ext = '.fraw'
	for m in range(len(timeBins) - 1):
		if _timeBins != None:
			addon = "_part" + str(m + 1) + "Of" + str(len(timeBins) - 1) 
		else:
			addon = ""
		groundTruth_motionStateDivide[m].swapaxes(2, 0).astype(\
								'float32').tofile(_baseNameOfOutput + addon + ext)




def readArguments():
	'''
	Def : Parse command line arguments.
	Return : Provided arguments.
	'''
	parser = argparse.ArgumentParser(description='Extract the ground truth of a binary '
		'file created by a Gate simulation. If time bins are given, extract an image '
		'for each time bins.')
	parser.add_argument('-iFile', dest='inputFile', type=str, required=True, \
	                    help='Path to the Gate file.')
	parser.add_argument('-dataType', dest='dataType', type=str, \
	                    required=True, choices=["basic_Gt", "basic_Gt_Inter"], \
	                    help='The data type used to save the coincidence in Gates.')
	parser.add_argument('-nbVoxel', dest='nbVoxel', type=int, nargs=3, \
	                    required=True, help='The number of voxels in each dimension.')	
	parser.add_argument('-imSize', dest='imSize', type=float, nargs=3, \
	                    required=True, help='The size of the image in each dimension.')	
	parser.add_argument('-oFile', dest='outputFile', type=str, required=True, \
	                    help='Path and basename of the file(s) where the ground ' \
	                         'truth(s) will be saved.')
	parser.add_argument('-timeBins', dest='timeBins', type=float, nargs='+', \
	                    required=False, default=None, help='The time bins used to ' \
	                    'segment the ground truth in multiple images.')	
	parser.add_argument('-cMode', dest='cMode', type=str, required=False,\
	                    choices=["ram", "part"], default='part',\
	                    help='Specify if the script should load all the data file in '\
	                    'the ram or only in part.')						
	
	return parser.parse_args()


#########################################################################################
# Main : We use this to make the main usable has a script and a method.
#########################################################################################
if __name__=='__main__':
	"""
	Currently, this module script functionality is only to extact the ground truth of a 
	binary file created by a Gate simulation.
	"""
	
	args = readArguments()

	if args.dataType == "basic_Gt":
 		dataType = "GateBinaryDataTypeWithSourcePos"
	elif  args.dataType == "basic_Gt_Inter":
		dataType = "GateBinaryDataTypeWithSourcePosAndGammaInter"
	else:
		print("The Gate binary data type " + str(args.dataType) + " is not defined.")

	extractGroundTruth(args.inputFile, dataType, args.nbVoxel, args.imSize, \
	                       args.outputFile, _timeBins=args.timeBins, _cMode=args.cMode)







