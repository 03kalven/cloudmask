import numpy as np
from PIL import Image
from netCDF4 import Dataset
import pandas as pd
import os
import scipy.signal


def all_three_only(in_img):
    """
    Returns an a filtered version of an input PIL image. Simple check for a bad pixel. Will be labeled -999 in the final cloud mask. Sets filtered pixels to (0,0,0).
    """
    img = np.array(in_img)
    img[
        ((img[..., 1] < 20) & (img[..., 2] < 20))
        | (img[..., 2] < 30)
        | (img[..., 0] == 0)
        | (img[..., 1] == 0)
        | (img[..., 2] == 0)
    ] = [0, 0, 0]
    return Image.fromarray(img)


def check_UNET_num(num):
    """
    Returns the output dimenstion for a UNET if the input number is a valid UNET input, or None if the input is not possible.
    """
    i = int(num)
    for k in range(4):
        i -= 4
        i /= 2
        if i != int(i):
            return 0
    for k in range(4):
        i -= 4
        i *= 2
    i -= 4
    return i


def check_UNET_num_back(num):
    """
    Returns the input dimenstion for a UNET if the input number is a valid UNET output, or None if the output is not possible.
    """
    i = int(num)
    for _ in range(4):
        i += 4
        i /= 2
        if i != int(i):
            return None
    for _ in range(4):
        i += 4
        i *= 2
    i += 4
    return i


def get_polar_bounds(ls, c=0):
    """
    Returns the southern and northen polar extents for a given solar longitude. Formulas provided by https://doi.org/10.1016/j.icarus.2019.05.041.

    Parameters:
    ls -- solar longitude (0 to 360)
    c -- degree offset. Moves the polar extents c degrees towards the center
    """
    lat_n = 59.0
    lat_s = -57.0

    if ls <= 90:
        lat_n = 59.0 + 0.214 * ls - c
    elif ls <= 180:
        lat_n = 59.0 + 0.214 * (180 - ls) - c
    elif ls <= 270:
        lat_s = -12.0 - 0.247 * ls + c
    elif ls <= 360:
        lat_s = -12.0 - 0.247 * (540 - ls) + c

    return (lat_n, lat_s)


def get_polar_bounds_new(ls, c=0):
    """
    Returns the southern and northen polar extents for a given solar longitude. Formulas derived from https://doi.org/10.1016/j.icarus.2019.05.041 and my additional independent analysis.

    Parameters:
    ls -- solar longitude (0 to 360)
    c -- degree offset. Moves the polar extents c degrees towards the center
    """
    if ls <= 180:
        lat_n = 59.0 + 0.214 * (90 - abs(90 - ls)) - c
        lat_s = -57.0 + 0.247 * (90 - abs(90 - ls)) + c
    else:
        lat_n = 59.0 - 0.214 * (90 - abs(270 - ls)) - c
        lat_s = -57.0 - 0.247 * (90 - abs(270 - ls)) + c

    return (lat_n, lat_s)


def bound_to_pixel(bound, size=1801, ax="lat"):
    """
    Converts the latitude/longitude degree value to a pixel value, 0 is top and size is bottom.

    Parameters:
    bound -- latitude or longitude value, in degrees,
    size --  pixels along the goven axis (default 1801)
    ax -- latitude ('lat') or longitude ('lon')
    """
    if ax == "lat":
        deg = 180
    elif ax == "lon":
        deg = 360

    res = deg / size
    mid = size // 2
    pix = round(mid - bound / res)

    return pix


def pixel_to_bound(pix, size=1801, ax="lat"):
    """
    Converts the pixel value to a latitude/longitude degree value.

    Parameters:
    bound -- latitude or longitude value, in degrees,
    size --  pixels along the goven axis (default 1801)
    ax -- latitude ('lat') or longitude ('lon')
    """
    if ax == "lat":
        deg = 180
    elif ax == "lon":
        deg = 360

    res = deg / size
    mid = size // 2
    bound = (pix - mid) * -1 * res

    return bound


def get_cloudmask_bounds(clouds):
    """
    Returns the upper and lower bounds of the cloud mask in pixels.

    Parameters:
    clouds -- numpy array of a given cloud mask
    """
    (full_y, _) = clouds.shape
    y = 0
    top = 0
    bottom = 0

    while y < full_y:
        if clouds[y, :].max() >= 0:
            top = y
            break
        y += 1

    y = full_y - 1
    while y >= 0:
        if clouds[y, :].max() >= 0:
            bottom = y
            break
        y -= 1
    return (top, bottom)


def get_cloudmask(mask_path):
    """
    Takes a .ncdf cloudmask path and returns its values in a 2D numpy array
    """
    cloudmask_ncdf = Dataset(mask_path)
    cloudmask = cloudmask_ncdf.variables["cloudmask"][:, :]
    cloudmask = np.flip(cloudmask, axis=0)
    return cloudmask


def binary_cloudmask(clouds):
    """
    Takes an array of cloud mask probabilities and returns a binary classification
    """
    cloudmask = clouds.copy()
    cloudmask[cloudmask < 0.5] = 0
    cloudmask[cloudmask >= 0.5] = 1
    return cloudmask


def get_black_bounds(img, sides=""):
    """
    Returns the first left and right bounds of an MDGM that are NOT entirely black. Uses 30 as a threshold for black pixel values instead of 0 in case of JPEG compression.

    Parameters:
    img -- PIL image of the RGB MDGM
    sides -- 'l' if want to crop left bound, 'r' if want to crop right bound, 'lr' if want to crop both, '' if want to crop neither (default '')
    """
    xsize, ysize = img.size

    xmin, xmax = 0, xsize

    np_img = np.array(img)
    filtered_img = np.zeros((ysize, xsize))
    filter = (np_img[..., 0] < 30) | (np_img[..., 1] < 30) | (np_img[..., 2] < 30)
    filtered_img[np.invert(filter)] = 1

    temp = np.sum(filtered_img, axis=0)
    temp[temp >= 1] = 1

    if "l" in sides:
        # crop left
        xmin = np.argmax(temp)
    if "r" in sides:
        # crop right
        xmax = temp.shape[0] - (1 + np.argmax(temp[::-1]))
    return (xmin, xmax)


def get_info_train(img_path):
    """
    Retrieves the solar longitude and martian year for a given photo located in the directory structure below. Cloudmasks retrieved from https://doi.org/10.7910/DVN/WU6VZ8 should satisfy this structure:

    B01
    │   list_ls.txt
    │
    └───cloudmask
    │   │   cloudmask_B01day01.ncdf
    │   │   cloudmask_B01day02.ncdf
    │   │   cloudmask_B01day03.ncdf
    │   │   ...
    │
    └───mdgms
        │   B01day01.jpeg
        │   B01day02.jpeg
        │   B01day03.jpeg
        │   ...
    """
    path = os.path.abspath(img_path)
    mdgm_split = os.path.split(path)
    subset_path = os.path.split(mdgm_split[0])[0]
    ls_path = os.path.join(subset_path, "list_ls.txt")
    subset_and_day = mdgm_split[1][:8]

    all_ls = pd.read_table(
        ls_path, delim_whitespace=True, names=["name", "my", "ls"], engine="python"
    )
    ls = float(all_ls[all_ls["name"] == "{}".format(subset_and_day)]["ls"])
    my = int(all_ls[all_ls["name"] == "{}".format(subset_and_day)]["ls"])
    return ls, my


def get_info_compute(img_path):
    """
    Retrieves the solar longitude and martian year for a given photo located in the directory structure below. Data retrieved from https://doi.org/10.7910/DVN/U3766S should satisfy this structure:

    B
    │
    └───B01
    │   │   B01_ls.txt
    │   │
    │   │   B01_day01_zequat.jpg
    │   │   B01_day02_zequat.jpg
    │   │   B01_day32_zequat.jpg
    │   │   ...
    │   │
    │   └───list
    │       │   B01_day01.list
    │       │   B01_day02.list
    │       │   B01_day03.list
    │       │   ...
    │
    └───B02
    │
    └───B03
    ...
    """
    path = os.path.abspath(img_path)
    subsetSplit = os.path.split(path)
    subset_and_day = subsetSplit[1][:9]
    ls_path = os.path.join(subsetSplit[0], "{}_ls.txt".format(subset_and_day[:3]))
    all_ls = pd.read_table(
        ls_path, delim_whitespace=True, names=["name", "my", "ls"], engine="python"
    )
    ls = float(all_ls[all_ls["name"] == subset_and_day]["ls"])
    my = int(all_ls[all_ls["name"] == subset_and_day]["ls"])
    return ls, my


def get_info_model(img_path):
    """
    Retrieves the solar longitude and martian year for a given photo/cloudmask pair located in the directory structure below. Cloudmasks generated by the model should satisfy this structure:

    B01
    │   B01_ls.txt
    │
    └───cloudmasks
    │   │   cloudmask_B01day01.ncdf
    │   │   cloudmask_B01day02.ncdf
    │   │   cloudmask_B01day03.ncdf
    │   │   ...
    │
    └───mdgms
        │   B01day01.jpeg
        │   B01day02.jpeg
        │   B01day03.jpeg
        │   ...
    """
    path = os.path.abspath(img_path)
    mdgm_split = os.path.split(path)
    subset_path = os.path.split(mdgm_split[0])[0]
    ls_path = os.path.join(subset_path, "{}_ls.txt".format(mdgm_split[1][:3]))
    subset_and_day = mdgm_split[1][:9]

    all_ls = pd.read_table(
        ls_path, delim_whitespace=True, names=["name", "my", "ls"], engine="python"
    )
    ls = float(all_ls[all_ls["name"] == subset_and_day]["ls"])
    my = int(all_ls[all_ls["name"] == subset_and_day]["my"])
    return ls, my


def get_cloudmask_train(img_path):
    """
    Retrieves the cloudmask for a given photo located in the directory structure below. Cloudmasks retrieved from https://doi.org/10.7910/DVN/WU6VZ8 should satisfy this structure:

    B01
    │   list_ls.txt
    │
    └───cloudmask
    │   │   cloudmask_B01day01.ncdf
    │   │   cloudmask_B01day02.ncdf
    │   │   cloudmask_B01day03.ncdf
    │   │   ...
    │
    └───mdgms
        │   B01day01.jpeg
        │   B01day02.jpeg
        │   B01day03.jpeg
        │   ...
    """
    path = os.path.abspath(img_path)
    mdgm_split = os.path.split(path)
    subset_path = os.path.split(mdgm_split[0])[0]
    subset_and_day = mdgm_split[1][:8]

    cloudmask_path = os.path.join(
        subset_path, "cloudmask", "cloudmask_{}.ncdf".format(subset_and_day)
    )
    return cloudmask_path


def get_cloudmask_predicted(img_path):
    """
    Retrieves the cloudmask for a given photo located in the directory structure below. Cloudmasks from this ML model should satisfy this structure:

    B01
    │   list_ls.txt
    │
    └───cloudmasks
    │   │   cloudmask_B01day_01.ncdf
    │   │   cloudmask_B01day_02.ncdf
    │   │   cloudmask_B01day_03.ncdf
    │   │   ...
    │
    └───mdgms
        │   B01day_01.jpeg
        │   B01day_02.jpeg
        │   B01day_03.jpeg
        │   ...
    """
    path = os.path.abspath(img_path)
    mdgm_split = os.path.split(path)
    subset_path = os.path.split(mdgm_split[0])[0]
    subset_and_day = mdgm_split[1][:9]

    cloudmask_path = os.path.join(
        subset_path, "cloudmasks", "cloudmask_{}.ncdf".format(subset_and_day)
    )
    return cloudmask_path


def pad_mdgm(mdgm, xhigh, xlow, yhigh, ylow):
    """
    Prepares an inputted mdgm for subdividing. Crops the image's black borders and polar extents out, and adds 92 pixels of each end of the mdgm to the opposite ends.

    Parameters:
    mdgm -- a PIL image
    xhigh & xlow -- the x coordinate for cropping (xhigh > xlow, 0 is left)
    yhigh & ylow -- the y coordinates for cropping (yhigh > ylow, 0 is top)
    """
    # for cropping, use xhigh + 1 and yhigh + 1 because that index is the first completely black column
    cropped_mdgm = mdgm.crop((xlow, ylow - 92, xhigh + 1, yhigh + 93))
    croppedw, croppedh = cropped_mdgm.size
    padded_mdgm = Image.new("RGB", (croppedw + 2 * 92, croppedh))

    (fullw, fullh) = padded_mdgm.size
    padded_mdgm.paste(
        cropped_mdgm.crop((croppedw - 92, 0, croppedw, fullh)), (0, 0, 92, fullh)
    )
    padded_mdgm.paste(cropped_mdgm, (92, 0, croppedw + 92, fullh))
    padded_mdgm.paste(
        cropped_mdgm.crop((0, 0, 92, fullh)), (fullw - 92, 0, fullw, fullh)
    )

    return padded_mdgm


def _spline_window(window_size, power=2):
    """
    {Credit: Vooban, https://github.com/Vooban/Smoothly-Blend-Image-Patches}
    Returns a spline function:

    Parameters:
    window_size -- the domain of the spline function [0, window_size]
    power -- power of the spline function (default 2)
    """
    intersection = int(window_size / 4)
    wind_outer = (abs(2 * (scipy.signal.triang(window_size))) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def window_2D(window_size, power=2):
    """
    {Credit: Vooban, https://github.com/Vooban/Smoothly-Blend-Image-Patches}
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.

    Parameters:
    window_size -- the domain of the spline function [0, window_size]
    power -- power of the spline function (default 2)
    """
    wind = _spline_window(window_size, power)
    wind = np.expand_dims(wind, 1)
    wind = wind * wind.transpose(1, 0)
    return np.expand_dims(wind, axis=0)
