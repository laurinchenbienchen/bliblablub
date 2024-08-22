import numpy as np


# function to calculate the mean squared error between to images
def compute_mse(image1, image2):
    # check if the images have the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same size")
    # compute mse: average of squared differences between images
    mse = np.mean((image1.astype("float") - image2.astype("float")) ** 2)
    return mse


# function to compute structural similarity index between to images
def fitness_ssim(image1, image2):
    # constant for stabilization in the ssim calculations
    c1 = 6.5025
    c2 = 58.5225

    # convert images to float64 type for more accurate calculations
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    # compute mean values and variances of the images
    mu1 = np.mean(image1)
    mu2 = np.mean(image2)
    sigma1_sq = np.var(image1)
    sigma2_sq = np.var(image2)
    sigma12 = np.cov(image1.ravel(), image2.ravel())[0, 1]

    # calculate ssim value
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim


# function to calculate the normalized cross correlation
def fitness_ncc(image1, image2):
    # compute mean values of the images
    mean1 = np.mean(image1)
    mean2 = np.mean(image2)
    # compute the numerator of the ncc
    numerator = np.sum((image1 - mean1) * (image2 - mean2))
    # compute the denominator of the ncc
    denominator = np.sqrt(np.sum((image1 - mean1) ** 2) * np.sum((image2 - mean2) ** 2))
    # calculate ncc and return the value
    ncc = numerator / denominator
    return ncc


# mutual information
def m_i(image1, image2, bins=20):
    # compute the 2d histo for the images
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)
    # normalize the histogram to obtain a probability distribution
    pxy = hist_2d / np.sum(hist_2d)
    # compute the marginal distributions for the two images
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    # compute the marginal distributions for the two images
    px_py = np.outer(px, py)

    # compute mutual information
    nzs = pxy > 0  # considering only the non-zero values
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    return mi


# compute peak signal-to-noise ratio between two images
def fitness_psnr(image1, image2):
    # compute the mean squared error between the images
    mse = np.mean((image1 - image2) ** 2)
    # if mse is 0 the images are identical. return a very high psnr value
    if mse == 0:
        return float('inf')
    # maximum pixel value typically used for 8-bit images
    max_pixel = 255.0
    # calculate psnr
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
