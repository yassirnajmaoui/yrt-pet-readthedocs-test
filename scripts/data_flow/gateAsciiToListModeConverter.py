#!/usr/bin/env python
import gateReader as gR
from savantDoiGateToListModeConverter import *
import argparse
import numpy as np

def convertGateDetIdToCastorDetId(_eventGateDetId, _sortProjDetId=True):
    '''
    This is a simpler version of convertGateDetIdToCastorDetId available in savantDoiGateToListModeConverter
    '''
    eventBasicDetId = convertGateDetIdToBasisDetId(_eventGateDetId)

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
    
    return eventCastorDetId

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Convert the ASCII output of a Gate file to a a YRT-PET-compatible ListMode file. (Works only for SAVANT or UHR)')
    parser.add_argument('-i', dest='inputFile', type=str, required=True, help='Path to the ASCII file.')
    parser.add_argument('-o', dest='outputFile', type=str, required=True, help='Path to the output ListMode file.')
    args = parser.parse_args()

    in_file = args.inputFile
    photonPairArrTime, photonPairGateDetId = gR.extractSimulDataFromGateTextFile(in_file)
    eventCastorDetId = convertGateDetIdToCastorDetId(photonPairGateDetId)
    lm = np.zeros([eventCastorDetId.shape[0],3],dtype=np.uint32)
    lm[:,1] = eventCastorDetId[:,0]
    lm[:,2] = eventCastorDetId[:,1]
    lm.tofile(args.outputFile)
