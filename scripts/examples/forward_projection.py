#!/usr/bin/env python
import pyyrtpet as gc

# Note: This file is not to be executed, but simply to be used as a documentation

scanner = gc.GCScannerOwned("<path to the scanner's json file>")

# Prepare an empty histogram
outHis = gc.GCHistogram3DOwned(scanner)
outHis.Allocate()

# Read an image to Forward-project
imgParams = gc.ImageParams("<path to the image parameters file>")
inputImage = gc.ImageOwned(imgParams, "<path to input image>")

projectorType = gc.GCOperatorProjector.ProjectorType.DD_GPU
# Available projectors: SIDDON, DD, DD_GPU
gc.forwProject(scanner, inputImage, outHis, projectorType)

outHis.writeToFile("<path to save output histogram>")
