#!/usr/bin/env python
import scipy as scp
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import pyyrtpet as gc

mpl.use("Qt5Agg")

pi=np.pi
fwhm_sigma_ratio = 2*np.sqrt(2*np.log(2))

def target_1d(count, mu, background, amplitude, fwhm):
    sigma = fwhm/fwhm_sigma_ratio
    m = np.arange(count)
    g = background+ amplitude*np.exp(-(m-mu)**2.0/(2*sigma**2.0))
    return g

def target(shape, mu_x, mu_y, background, amplitude, fwhm_x, fwhm_y):
    sigma_x = fwhm_x/fwhm_sigma_ratio
    sigma_y = fwhm_y/fwhm_sigma_ratio
    my, mx = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    g = background + amplitude*np.exp(-(mx-mu_x)**2.0/(2*sigma_x**2.0)-(my-mu_y)**2.0/(2*sigma_y**2.0))
    return g.reshape([-1]) # Return model, flatten for compatibility with curve_fit

def target_fixed(fixed_args, fwhm_x, fwhm_y):
    return target((fixed_args[0],fixed_args[1]),fixed_args[2],fixed_args[3],fixed_args[4],fixed_args[5],fwhm_x,fwhm_y)

def fit_img(img: np.ndarray, disp=False):
    mu_x_init, mu_y_init = np.unravel_index(np.argmax(img), img.shape)
    fwhm_init = 1
    initial_vals = [mu_x_init, mu_y_init, np.min(img), np.max(img), fwhm_init, fwhm_init]
    bounds_min = (0,0,0,0,0,0)
    bounds_max = (img.shape[1], img.shape[0], np.max(img), np.inf, img.shape[1], img.shape[0])
    if(disp):
        plt.matshow(img)
        plt.show()
        plt.matshow(target(img.shape, *initial_vals).reshape(img.shape))
        plt.show()

    popts, pcov = scp.optimize.curve_fit(target, img.shape, img.reshape([-1]), p0=initial_vals, bounds=(bounds_min,bounds_max))
    if(disp):
        plt.plot(img[mu_y_init,:])
        plt.plot(target_1d(img.shape[1], popts[1], popts[2], popts[3], popts[5]))
        plt.show()

    return popts

def fit_img_fixed(img: np.ndarray, *args):
    fwhm_init = 1
    initial_vals = (fwhm_init, fwhm_init)
    bounds_min = (0,0)
    bounds_max = (img.shape[1], img.shape[0])

    popts, pcov = scp.optimize.curve_fit(target_fixed, np.array([*args]), img.reshape([-1]), p0=initial_vals, bounds=(bounds_min,bounds_max), check_finite=False)

    return popts


if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Data processing')
    parser.add_argument('--image', type=str, nargs='+', required=True, help='YRT-PET image file(s)')
    parser.add_argument('--slice_list', type=str, required=True, help='CSV file of the list of slices to consider')
    parser.add_argument('--crop_box_list', type=str, required=True, help='CSV file of the list of crop-boxes')
    parser.add_argument('--res_full', type=str, required=False, help='res_full.npy file if already computed')
    args = parser.parse_args()

    # slices to average on
    slice_list = np.genfromtxt(args.slice_list, delimiter=',', dtype=np.uint64) # format = s1,s2,s3,s4,...
    if(slice_list.ndim==0):
        slice_list = np.array([slice_list])
    crop_box_list = np.genfromtxt(args.crop_box_list, delimiter=',', dtype=np.uint64) # format: x1,y1,x2,y2 \n x1,y1,x2,y2

    num_boxes = len(crop_box_list)
    num_files = len(args.image)

    if(args.res_full is None):
        out_data = np.zeros([num_files, num_boxes, 6])

        for i_img_fname in tqdm(range(num_files)):
            img_fname = args.image[i_img_fname]
            gc_img = gc.Array3Ddouble()
            gc_img.ReadFromFile(img_fname)
            img = np.array(gc_img, copy=False)
            img_z_averaged = np.mean(img[slice_list],axis=0)
            
            for i_box in range(num_boxes):
                box = crop_box_list[i_box]
                box_x = min(box[0],box[2]), max(box[0],box[2])
                box_y = min(box[1],box[3]), max(box[1],box[3])
                cropped_img = img_z_averaged[box_y[0]:box_y[1], box_x[0]:box_x[1]]
                popts = fit_img(cropped_img)
                out_data[i_img_fname, i_box] = [*popts]
        
        np.save("res_full", out_data)
        res = out_data
    else:
        res = np.load(args.res_full)

    print("Re-fitting all the iterations by fixing all the parameters with the mean of the previous fittings, except for fwhms")
    out_data_fixed = np.zeros([num_files,num_boxes,2])
    mu_x = np.mean(res[:,:,0],axis=0)
    mu_y = np.mean(res[:,:,1],axis=0)
    background = np.mean(res[:,:,2],axis=0)
    amplitude = np.mean(res[:,:,3],axis=0)

    for i_img_fname in tqdm(range(num_files)):
        img_fname = args.image[i_img_fname]
        gc_img = gc.Array3Ddouble()
        gc_img.ReadFromFile(img_fname)
        img = np.array(gc_img, copy=False)
        img_z_averaged = np.mean(img[slice_list],axis=0)

        for i_box in range(num_boxes):
            box = crop_box_list[i_box]
            box_x = min(box[0],box[2]),max(box[0],box[2])
            box_y = min(box[1],box[3]),max(box[1],box[3])
            cropped_img = img_z_averaged[box_y[0]:box_y[1], box_x[0]:box_x[1]]
            popts = fit_img_fixed(cropped_img, cropped_img.shape[0], cropped_img.shape[1], mu_x[i_box], mu_y[i_box], background[i_box], amplitude[i_box])
            out_data_fixed[i_img_fname, i_box] = [*popts]

    np.save("res_full_fixed", out_data_fixed)
    res = out_data_fixed

    for i_crop_box in range(len(crop_box_list)):
        plt.scatter(np.arange(res.shape[0]),np.sqrt(res[:,i_crop_box,0]**2+res[:,i_crop_box,1]**2))
        plt.show()
