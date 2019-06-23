from os import listdir
from utils.imgutils import segment, normalize
from cv2 import imread
from multiprocessing import Pool, cpu_count
from itertools import repeat
from fnmatch import filter
import numpy as np
import scipy.io as sio
import os
import warnings
warnings.filterwarnings("ignore")

##########################################################################
#  Function which generate the iris template using in the matching
##########################################################################
DATABASE_PATH = './templates/'


def encode_iris(arr_polar, arr_noise, minw_length, mult, sigma_f):
    """
    Generate iris template and noise mask from the normalised iris region.
    """
    # convolve with gabor filters
    filterb = gaborconvolve_f(arr_polar, minw_length, mult, sigma_f)
    l = arr_polar.shape[1]
    template = np.zeros([arr_polar.shape[0], 2 * l])
    h = np.arange(arr_polar.shape[0])

    # making the iris template
    mask_noise = np.zeros(template.shape)
    filt = filterb[:, :]

    # quantization and check to se if the phase data is useful
    H1 = np.real(filt) > 0
    H2 = np.imag(filt) > 0

    H3 = np.abs(filt) < 0.0001
    for i in range(l):
        ja = 2 * i

        # biometric template
        template[:, ja] = H1[:, i]
        template[:, ja + 1] = H2[:, i]
        # noise mask_noise
        mask_noise[:, ja] = arr_noise[:, i] | H3[:, i]
        mask_noise[:, ja + 1] = arr_noise[:, i] | H3[:, i]

    return template, mask_noise


def gaborconvolve_f(img, minw_length, mult, sigma_f):
    """
    Convolve each row of an imgage with 1D log-Gabor filters.
    """
    rows, ndata = img.shape
    logGabor_f = np.zeros(ndata)
    filterb = np.zeros([rows, ndata], dtype=complex)

    radius = np.arange(ndata/2 + 1) / (ndata/2) / 2
    radius[0] = 1

    # filter wavelength
    wavelength = minw_length

    # radial filter component 
    fo = 1 / wavelength
    logGabor_f[0: int(ndata/2) + 1] = np.exp((-(np.log(radius/fo))**2) /
                                    (2 * np.log(sigma_f)**2))
    logGabor_f[0] = 0

    # convolution for each row
    for r in range(rows):
        signal = img[r, 0:ndata]
        imagefft = np.fft.fft(signal)
        filterb[r, :] = np.fft.ifft(imagefft * logGabor_f)
    
    return filterb


##########################################################################
# Function to extract the feature for the matching process
##########################################################################
def extractFeature(img_filename, eyelashes_threshold=80, multiprocess=True):
    """
    Extract features from an iris image
    """
    # parameters
    eyelashes_threshold = 80
    radial_resolution = 20
    angular_resolution = 240
    minw_length = 18
    mult = 1
    sigma_f = 0.5

    #  segmentation
    im = imread(img_filename, 0)
    ciriris, cirpupil, imwithnoise = segment(im, eyelashes_threshold,
                                    multiprocess)

    # normalization
    arr_polar, arr_noise = normalize(imwithnoise, ciriris[1],  ciriris[0], ciriris[2],
                                         cirpupil[1], cirpupil[0], cirpupil[2],
                                         radial_resolution, angular_resolution)

    #  feature encoding
    template, mask_noise = encode_iris(arr_polar, arr_noise, minw_length, mult,
    sigma_f)
    

    return template, mask_noise, img_filename


##########################################################################
# Functions that do the matching between the image and the 
# account/template
##########################################################################


def matchingTemplate(template_extr, mask_extr, template_dir, threshold=0.38):
    """
    Matching the template of the image with the ones in the database
    """
    # n# of accounts in the database
    n_files = len(filter(listdir(template_dir), '*.mat'))
    if n_files == 0:
        return -1

    # use every cores to calculate Hamming distances
    args = zip(
        sorted(listdir(template_dir)),
        repeat(template_extr),
        repeat(mask_extr),
        repeat(template_dir),
    )
    with Pool(processes=cpu_count()) as pools:
        result_list = pools.starmap(matchingPool, args)

    filenames = [result_list[i][0] for i in range(len(result_list))]
    hm_dists = np.array([result_list[i][1] for i in range(len(result_list))])

    # Removing NaN elements
    ind_valid = np.where(hm_dists > 0)[0]
    hm_dists = hm_dists[ind_valid]
    filenames = [filenames[idx] for idx in ind_valid]

    ind_thres = np.where(hm_dists <= threshold)[0]
    if len(ind_thres) == 0:
        return 0
    else:
        hm_dists = hm_dists[ind_thres]
        filenames = [filenames[idx] for idx in ind_thres]
        ind_sort = np.argsort(hm_dists)
        return [filenames[idx] for idx in ind_sort]


def HammingDistance(template1, mask1, template2, mask2):
    """
    Calculate the Hamming distance between two iris templates.
    """
    hd = np.nan

    # Shifting template left and right, use the lowest Hamming distance
    for shifts in range(-8, 9):
        template1s = shiftbits_ham(template1, shifts)
        mask1s = shiftbits_ham(mask1, shifts)

        mask = np.logical_or(mask1s, mask2)
        nummaskbits = np.sum(mask == 1)
        totalbits = template1s.size - nummaskbits

        C = np.logical_xor(template1s, template2)
        C = np.logical_and(C, np.logical_not(mask))
        bitsdiff = np.sum(C == 1)

        if totalbits == 0:
            hd = np.nan
        else:
            hd1 = bitsdiff / totalbits
            if hd1 < hd or np.isnan(hd):
                hd = hd1

    return hd


def shiftbits_ham(template, noshifts):
    """
    Shift the bit-wise iris patterns.
    """
    templatenew = np.zeros(template.shape)
    width = template.shape[1]
    s = 2 * np.abs(noshifts)
    p = width - s

    if noshifts == 0:
        templatenew = template

    elif noshifts < 0:
        x = np.arange(p)
        templatenew[:, x] = template[:, s + x]
        x = np.arange(p, width)
        templatenew[:, x] = template[:, x - p]

    else:
        x = np.arange(s, width)
        templatenew[:, x] = template[:, x - s]
        x = np.arange(s)
        templatenew[:, x] = template[:, p + x]

    return templatenew


def matchingPool(file_temp_name, template_extr, mask_extr, template_dir):
    """
    Perform matching session within a Pool of parallel computation
    """
    data_template = sio.loadmat('%s%s' % (template_dir, file_temp_name))
    template = data_template['template']
    mask = data_template['mask']

    # the Hamming distance
    hm_dist = HammingDistance(template_extr, mask_extr, template, mask)
    return (file_temp_name, hm_dist)
