#!/usr/bin/env python
import pyyrtpet as yrt

# Note: This file is not to be executed, but simply to be used as a documentation

scanner = yrt.ScannerOwned("<path to the scanner's json file>")

# Read an histogram
dataset = yrt.Histogram3DOwned(scanner)
dataset.readFromFile("<path to an histogram to be backprojected>")
# or, alternatively, read a ListMode file
dataset = yrt.ListModeLUTOwned(scanner)
dataset.readFromFile("<path to a listmode to be backprojected>")

# Prepare an image
imgParams = yrt.ImageParams("<path to the image parameters file>")
outImage = yrt.ImageOwned(imgParams)
outImage.allocate()

projectorType = yrt.OperatorProjector.ProjectorType.DD_GPU
# Available projectors: SIDDON, DD, DD_GPU
yrt.backProject(scanner, outImage, dataset, projectorType)

outImage.writeToFile("<path where to save the output image>")
