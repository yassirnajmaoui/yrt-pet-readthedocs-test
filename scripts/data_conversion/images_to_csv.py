#!/bin/env python
import pylab as pl
from scipy.optimize import curve_fit
from scipy import asarray
from math import sqrt, log
import numpy as np
import argparse




if(__name__ == "__main__"):
    
    parser = argparse.ArgumentParser(description='Extracting values from a thin line of a Binary 3D array and export to CSV')
    parser.add_argument('--image', metavar='i', type=str,
                       help='filename of the image')
    parser.add_argument('--type', metavar='t', type=str,
                       help='data type of the bitmap (float64 or float32) default: float64')
    parser.add_argument('--dims', metavar='d', type=int,nargs=3,
                       help='dimensions of the image (Z,Y,X). if unspecified, it will read the header of the image')
    parser.add_argument('--coordinates', metavar='c', type=int,nargs=2,
                       help='coordinates in Z-Y for which line to check. default=2 10')
    args = parser.parse_args()
    
    exportimage = args.image+".csv"
    print("INFO: The exported image name will be " + exportimage)
    
    myoffset=0
    
    if(args.dims is None):
        imageheader=np.fromfile(args.image, count=8, dtype=np.int32)#read first 32 bytes
        dimensions=[imageheader[2],imageheader[4], imageheader[6]]#get dimensions z,y,x
        myoffset=32
    else:
        dimensions=args.dims
    #print(dimensions)
    
    if(args.type=="float64"):
        imageraw=np.fromfile(args.image, offset=myoffset, dtype=np.float64)
    elif(args.type=="float32"):
        imageraw=np.fromfile(args.image, offset=myoffset, dtype=np.float32)
    else:
        imageraw=np.fromfile(args.image, offset=myoffset, dtype=np.float64)
    
    image=imageraw.reshape(dimensions)
    
    coord1=2
    coord2=10
    if(args.coordinates is not None):
        coord1=args.coordinates[0]
        coord2=args.coordinates[1]
    datapoints = image[coord1][coord2]
    
    np.savetxt(exportimage, datapoints, delimiter=",")
    
    pl.plot(datapoints)
    pl.show()







