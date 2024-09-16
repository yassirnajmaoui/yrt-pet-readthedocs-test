#!/usr/bin/env python
import pyyrtpet as gc

# Note: This file is not to be executed, but simply to be used as a documentation

scanner = gc.GCScannerOwned("<path to the scanner's json file>")

# Read an histogram
dataset = gc.Histogram3DOwned(scanner)
dataset.readFromFile("<path to an histogram to be backprojected>")
# or, alternatively, read a ListMode file
dataset = gc.GCListModeLUTOwned(scanner)
dataset.readFromFile("<path to a listmode to be backprojected>")

# Prepare an image
imgParams = gc.ImageParams("<path to the image parameters file>")
outImage = gc.ImageOwned(imgParams)
outImage.Allocate()

projectorType = gc.GCOperatorProjector.ProjectorType.DD_GPU
# Available projectors: SIDDON, DD, DD_GPU
gc.backProject(scanner, inputImage, outHis, projectorType)

outImage.writeToFile("<path where to save the output image>")
