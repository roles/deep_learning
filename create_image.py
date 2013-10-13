#!/usr/bin/python

import PIL.Image
import sys
import numpy

def create_image(filename, nrow, ncol):
    f = open(filename)
    arr = []
    for line in f:
        arr.append([int(float(x) * 255) for x in line.split()])
    arr = numpy.array(arr).astype('uint8')
    image = PIL.Image.fromarray(arr, "L")
    image.save("test.png")

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print "invaild arguments"
    else:
        create_image(sys.argv[1], 28, 28)
