#!/usr/bin/python

import PIL.Image
import sys
import numpy


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def create_image(img_data, nrow, ncol, outfile, scale = True):
    space = 1
    N = img_data.shape[0]
    img_H = img_data.shape[1]
    img_W = img_data.shape[2]
    if(scale):
        for i in range(N):
            img_data[i] = scale_to_unit_interval(img_data[i])
    out_shape = numpy.zeros((nrow * img_H + nrow - 1, ncol * img_W + ncol - 1), dtype = "uint8")
    for i in range(nrow):
        for j in range(ncol):
            out_shape[(img_H + 1) * i : (img_H + 1) * i + img_H,
                      (img_W + 1) * j : (img_W + 1) * j + img_W] = img_data[i * ncol + j] * 255
    image = PIL.Image.fromarray(out_shape)
    image.save(outfile)

def print_W_image_from_file(filename):
    f = open(filename)
    img_H = 28
    for epcho in range(15):
        img_data = []
        for i in range(100):
            arr = []
            for j in range(img_H):
                line = f.readline()
                arr.append([float(x) for x in line.split()])
            img_data.append(arr)
        create_image(numpy.array(img_data), 10, 10, "../result/weight_epcho_%d.png" % epcho)

def print_sample_image_from_file(filename):
    f = open(filename)
    img_H = 28
    n_sample = 10
    n_node = 20
    img_data = []
    for i in range(n_sample):
        for k in range(n_node):
            arr = []
            for j in range(img_H):
                line = f.readline()
                arr.append([float(x) for x in line.split()])
            img_data.append(arr)
    create_image(numpy.array(img_data), n_sample, n_node, "../result/sample.png", False)

def print_da_weight_from_file(filename="da_weight.txt"):
    f = open(filename)
    img_H = 28
    img_data = []
    for i in range(100):
        arr = []
        line = f.readline()
        for j in range(img_H):
            line = f.readline()
            arr.append([float(x) for x in line.split()])
        img_data.append(arr)
    create_image(numpy.array(img_data), 10, 10, "da_weight_corrupt.png")

if __name__ == "__main__":
    #print_da_weight_from_file("da_weight_corrupt.txt")
    #print_W_image_from_file("../result/mnist_rbm_weight.txt")
    print_sample_image_from_file("../result/rbm_sample.txt")
    #create_image(sys.argv[1], 28, 28)
