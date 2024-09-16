#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module used to convert a file of positions over time to a file of warp parameters 
relative to a reference frame.

TODO:
	- Make it a class for easy implementation of other input file type.
"""


#########################################################################################
# Imports
import numpy as _np
import sys as _sys
import argparse as _argparse
import copy as _copy


#########################################################################################
# List of methods created to build the features.
def extractPositionParams(_inputFile):
	"""
	Def.: Extact the positions parameters from a file. 
	@_inputFile (String): Path to the file holding the position parameters.
	Note: Currently, this method only work with seconds, degrees and millimeters.
	TODO: 
		- Make it work with other type of parameters units.
		- Check for multiple instance of hyper parameter? 
		- Check if they are all defined.
	Return: (2D numpy array, Float)[:, 8]
	"""
	defaultHyperParam = {"Time": 's', "Rotation": 'deg', "Translation": 'mm'}
	
	motionParam = []
	
	for line in open(_inputFile):
		# Ignore comments '#' and empty line.
		if (not line.startswith("#")) and (not line.isspace()):
			# Other line should either define a motion frame or a unit type.
			if line.startswith(tuple(defaultHyperParam.keys())):
				parsedLine = line.split()
				if parsedLine[1] != defaultHyperParam[parsedLine[0]]:
					_sys.exit("Script only work with default parameters type. Exiting.")
			else:
				dataSplitted = line.split()
				if len(dataSplitted) != 8:
					_sys.exit("Incorrect number of position parameters was " + \
					          "extracted from \n \t \"" + line.strip() + "\"")
				motionParam.append([float(i) for i in line.split()])
				motionParam[-1][1] *= -1.0
				# For some reasons, the result of rotation using genericMove is the 
				# inverted of what the user specify, so we invert the angle.						
				
	return _np.array(motionParam)
			

def computeFrameTimeBins(_positionTimeInfo, _scanTimeInfo):		
	"""
	Def.: Compute the time bins of the position frame.
	Note: Mostly usefull for Gate placement file since the time specified for the 
		position does not correspond directly to time bins.
	@_positionTimeInfo (1D numpy array, Float)[M]: The time tag given to a position 
		frame.
	@_scanTimeInfo (List of 3 Float): ['Start of the scan', 'End of the scan', 'time 
		discretization of the scan']
	Return: (1D numpy array, Float)[M + 1]
	"""
	
	scanTimeDisc = _np.arange(_scanTimeInfo[0], _scanTimeInfo[1], _scanTimeInfo[2])
	# We also want the end time of the scan.
	if scanTimeDisc[-1] != _scanTimeInfo[1]:
		scanTimeDisc = _np.append(scanTimeDisc, _scanTimeInfo[1])
	
	frameTimeBins = _np.empty(_positionTimeInfo.size + 1)
	frameTimeBins[0] = scanTimeDisc[0]
	frameTimeBins[-1] = scanTimeDisc[-1]
	
	for cFrameId in range(0, _positionTimeInfo.size - 1):
		cFrameStart = _np.mean(_positionTimeInfo[cFrameId:(cFrameId + 2)])
		# In Gate, the current position is defined by which position time bins the start  
		# of the current time slice falls in.
		frameTimeBins[cFrameId + 1] = scanTimeDisc[_np.argmax(cFrameStart \
		                                                       < scanTimeDisc)]

	return frameTimeBins
			
			
def computeFrameWeight(_frameTimeBins, _srcHalfLife):	
	"""
	Def.: Compute the ratio of counts expected for each frame.
	@_frameTimeBins (1D numpy array, Float): The fame time bins.
	@_srcHalfLife (Float): The half-life, in seconds, of the source.
	Return: (1D numpy array, Float)
	"""
	if _srcHalfLife == 0.0:
		frameTimeLenght = _np.diff(_frameTimeBins)
		frameWeight = frameTimeLenght / frameTimeLenght.sum()
	else:
		frameWeight = _np.diff(0.5**(_frameTimeBins[::-1] / _srcHalfLife))[::-1]
		# Normalize to one.
		frameWeight = frameWeight / frameWeight.sum()
	
	return frameWeight
			
			
def convertPositionToWarpRelToRefFrame(_positionParam, _refFrameId):
	"""
	Def.: Convert the positions parameters to warping parameters relative to the 
		reference frame position.
	@_positionParam (2D numpy array, Float)[:, 7]: The position parameters of each 
		frame.
	@_refFrameId (Integer): The frame to be used as the referance frame.
	Return: (2D numpy array, Float)[:, 7]

	"""
	positionParamQuaternion = convertAxisAngleToQuaternion(_positionParam[:, :4])
	
	# frameQuaternion = warpQuaternion * refQuaternion => 
	#		frameQuaternion * conj(refQuaternion) = warpQuaternion
	refQuaternion = positionParamQuaternion[_refFrameId, :]
	refConjQuaternion = _copy.deepcopy(refQuaternion)
	refConjQuaternion[1:] *= -1.0
	warpQuaternion = quaternionMultiply(refConjQuaternion, positionParamQuaternion)
	
	refTranslation = _positionParam[_refFrameId, 4:]
	refTranslationRotated = applyQuaternionRotationTo3Dpoint(warpQuaternion, \
	                                                         refTranslation)
	
	warpTranslation = _positionParam[:, 4:] - refTranslationRotated

	frameWarpParam = _np.zeros(_positionParam.shape)
	# Set the rotation parameters for the warp.
	frameWarpParam[:, :4] = warpQuaternion
	# Set the translation parameters for the warp.
	frameWarpParam[:, 4:] = warpTranslation
	
	return frameWarpParam
		

def writeWarpParamFile(_outputFileName, _refFrameId, _frameTimeBins, _frameWeight, \
                       _frameWarpParam, _inputFileName):
	"""
	Def.: Create the warp parameters file.
	@_outputFileName (String): The path and basename of the warp parameters file.
	@_refFrameId (Integer): Reference frame Id.
	@_frameTimeBins (1D numpy array, Float): The time bins of the motion frame.
	@_frameWeight (1D numpy array, Float): The ratio of counts expected for each frame.
	@_frameWarpParam (2D numpy array, Float)[:, 7]: The warping parameters of each frame.
	@_inputFileName (String): The file which was converted to the warp parameters file.
	"""
	fileType = 't'
	fileExt = fileType + 'wp'
	
	wpFileName = _outputFileName + "." +fileExt
	wpFile = open(wpFileName, "w") 
	
	wpFile.write("# Created from the Gate placement file " + str(_inputFileName) + "\n")
	wpFile.write("# Units used: Time s, translation mm\n")
	
	wpFile.write("Reference frame Id: " + str(_refFrameId) + "\n")

	# For some reason, I was not able to deactive max_line_width. So, I used 1000 to 
	# make sure that it would all fit into one line.
	for m in range(_frameTimeBins.size - 1):
		wpFile.write(str(round(_frameTimeBins[m], 5)) + " " + str(_frameWeight[m]) \
		              + " " + _np.array2string(_frameWarpParam[m, :], precision=10, 
		              separator=' ', max_line_width=1000)[1:-1] + "\n")
	
	wpFile.close() 
	
	
#########################################################################################
# List of tools used in this module.
def convertAxisAngleToQuaternion(_axisAngle):
	"""
	Def.: Convert an axis-angle orientation to quaternion orientation.
	@_axisAngle (1-2D numpy array, Float)[(:), 4]: Axis-angle orientation with the first 
		element being the angle in degree.
	Return: (2D numpy array, Float)[:, 4]
	"""
	# To make it work with _axisAngle in 1D and 2D.
	if _axisAngle.ndim == 1:
		_axisAngle = _axisAngle[_np.newaxis, :]	
	
	# Make sure it is normalized.
	_axisAngle[:, 1:] /= _np.linalg.norm(_axisAngle[:, 1:], axis=1)[:, _np.newaxis]
	angleRadian = _axisAngle[:, 0] / 180.0 * _np.pi
	
	qPositionParam = _np.zeros(_axisAngle.shape)
	qPositionParam[:, 0] = _np.cos(angleRadian / 2.0)
	qPositionParam[:, 1] = _axisAngle[:, 1] * _np.sin(angleRadian / 2.0)
	qPositionParam[:, 2] = _axisAngle[:, 2] * _np.sin(angleRadian / 2.0)
	qPositionParam[:, 3] = _axisAngle[:, 3] * _np.sin(angleRadian / 2.0)
	
	return _np.squeeze(qPositionParam)
	

def quaternionMultiply(_lQuat, _rQuat):
	"""
	Def.: Multiply the quanternions in _lQuat, stored line-wise, to the quaternion 
		_rQuat.
	@_lQuat (1-2D numpy array, Float)[(:), 4]: Quaternion(s) multiplied to the _rQuat.
	@_rQuat (1D numpy array, Float)[4]: A quaternion.
	Return: (2D numpy array, Float)[:, 4]
	"""
	# To make it work with _lQuat in 1D and 2D.
	if _rQuat.ndim == 1:
		_rQuat = _rQuat[_np.newaxis, :]
		
	resultingQuaternion = _np.zeros((_rQuat.shape[0], 4))
	
	resultingQuaternion[:, 0] = (_lQuat[0] * _rQuat[:, 0]) - (_lQuat[1] * _rQuat[:, 1])\
	                            - (_lQuat[2] * _rQuat[:, 2]) - (_lQuat[3] * _rQuat[:, 3])
								
	resultingQuaternion[:, 1] = (_lQuat[0] * _rQuat[:, 1]) + (_lQuat[1] * _rQuat[:, 0])\
	                            + (_lQuat[2] * _rQuat[:, 3]) - (_lQuat[3] * _rQuat[:, 2])
								
	resultingQuaternion[:, 2] = (_lQuat[0] * _rQuat[:, 2]) - (_lQuat[1] * _rQuat[:, 3])\
	                            + (_lQuat[2] * _rQuat[:, 0]) + (_lQuat[3] * _rQuat[:, 1])
								
	resultingQuaternion[:, 3] = (_lQuat[0] * _rQuat[:, 3]) + (_lQuat[1] * _rQuat[:, 2])\
	                            - (_lQuat[2] * _rQuat[:, 1]) + (_lQuat[3] * _rQuat[:, 0])

	return _np.squeeze(resultingQuaternion)
	

def applyQuaternionRotationTo3Dpoint(_qRots, _3DPoint):
	"""
	Def.: Apply the rotations defined in _qRots to the 3D position _3DPoint.
	@_qRots (1-2D numpy array, Float)[(:), 4]: Rotations, one in each line, defined as 
		quaternion.
	@_3DPoint (1D numpy array, Float)[3]: 3D point to rotate.
	Code inspired from:
	https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
	Idea:
	Let u = (q.x, q.y, q.z); s = q.w
	Then v rotated by q :=  2.0f * dot(u, v) * u 
	                        + (s*s - dot(u, u)) * v
                            + 2.0f * s * cross(u, v);
	"""
	# To make it work with _qRots in 1D and 2D.
	if _qRots.ndim == 1:
		_qRots = _qRots[_np.newaxis, :]
	
	firstPart = 2.0 * _np.sum(_qRots[:, 1:] * _3DPoint, axis=1)[:, _np.newaxis] \
	            * _qRots[:, 1:]
	secondPart = (_qRots[:, 0]**2 - (_qRots[:, 1:] **2).sum(axis=1))[:, _np.newaxis] \
	             * _3DPoint
	thirdPart =  2.0 * _qRots[:, 0][:, _np.newaxis] * _np.cross(_qRots[:, 1:], _3DPoint)
	
	rotated3DPoints = firstPart + secondPart + thirdPart
	
	return _np.squeeze(rotated3DPoints)


#########################################################################################
# List of methods used to use this module as a script.


def readArguments():
	'''
	Def : Parse command line arguments.
	Return : Provided arguments.
	'''
	parser = _argparse.ArgumentParser(description='Convert a positions file into a '
		'warping parameters file used for motion corrected reconstruction.')
	parser.add_argument('-iFile', dest='inputFile', type=str, required=True, \
						help='Path to the file holding the positions.')
	parser.add_argument('-oFile', dest='outputFile', type=str, required=True, \
						help='Path to where the warping parameters will be saved.')
	parser.add_argument('-iFileType', dest='inputFileType', type=str, required=True, \
						choices=['gate'], \
						help='The type of position file.')
	parser.add_argument('-scanTimeInfo', dest='scanTimeInfo', type=float, \
						required=True, nargs=3, help='The start, end and sampling of '
						'the tracking file.')
	parser.add_argument('-refFrameId', dest='refFrameId', required=True,\
						type=int, help='Which frame will be used as reference.')
	parser.add_argument('-srcHalfLife', dest='srcHalfLife', required=False,\
						type=float, default=0.0, \
						help='The half-life, in seconds, of the source.')
	
	return parser.parse_args()




#########################################################################################
# Main : We use this to make the main usable has a script and a method.
#########################################################################################

if __name__== "__main__":
	"""
	Currently, this module cannot be used as a script. Only used to test feature.
	# python3 ~/workGit/highrespetrecon/Script/DataFlow/motionWarpParametersExtractor.py 
		-iFile GateSimulation/MacFiles/piston.placements -refFrameId 0 
		-oFile warpParam/piston.wpf -scanTimeInfo 0.0 10.0 0.001 -iFileType gate
	"""

	args = readArguments()
	
	if args.inputFileType == "gate":
		motionParam = extractPositionParams(args.inputFile)
	else:
		sys.exit("The input file type " + str(args.inputFileType) + " is not " \
		         "supported. Exiting.")
	
	frameTimeBins = computeFrameTimeBins(motionParam[:, 0], args.scanTimeInfo)
	
	frameWeight = computeFrameWeight(frameTimeBins, args.srcHalfLife)
	
	frameWarpParam = convertPositionToWarpRelToRefFrame(motionParam[:, 1:], \
	                                                    args.refFrameId)
	writeWarpParamFile(args.outputFile, args.refFrameId, frameTimeBins, frameWeight, \
	                   frameWarpParam, args.inputFile)
	