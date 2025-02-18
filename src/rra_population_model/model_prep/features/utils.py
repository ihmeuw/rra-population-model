from typing import Literal

import numpy as np
import numpy.typing as npt
import rasterra as rt
from scipy.signal import oaconvolve


def precise_floor(a: float, precision: int = 0) -> float:
    """Round a number down to a given precision.

    Parameters
    ----------
    a
        The number to round down.
    precision
        The number of decimal places to round down to.

    Returns
    -------
    float
        The rounded down number.
    """
    return float(np.true_divide(np.floor(a * 10**precision), 10**precision))


def suppress_noise(
    raster: rt.RasterArray,
    noise_threshold: float = 0.01,
    fill_value: float = 0.0,
) -> rt.RasterArray:
    """Suppress small values in a raster.

    Parameters
    ----------
    raster
        The raster to suppress noise in.
    noise_threshold
        The threshold below which values are considered noise.

    Returns
    -------
    rt.RasterArray
        The raster with small values suppressed
    """
    raster._ndarray[raster._ndarray < noise_threshold] = fill_value  # noqa: SLF001
    return raster


def make_smoothing_convolution_kernel(
    pixel_resolution_m: int | float,
    radius_m: int | float,
    kernel_type: Literal["uniform", "gaussian"] = "uniform",
) -> npt.NDArray[np.float64]:
    """Make a convolution kernel for spatial averaging/smoothing.

    A convolution kernel is a (relatively) small matrix that is used to apply a
    localized transformation to a raster. Here we are choosing a kernel whose
    values are all positive and sum to 1 (thus representing a probability mass
    function). This special property means that the kernel can be used to
    compute a weighted average of the pixels in a neighborhood of a given pixel.
    In image processing, this is often used to smooth out noise in the image or
    to blur the image.

    This function produces both uniform and gaussian kernels. A uniform kernel
    is a circle with equal weights for all pixels inside the circle. A gaussian
    kernel is a circle with a gaussian distribution of weights, i.e. the weights
    decrease as you move away from the center of the circle.

    Parameters
    ----------
    pixel_resolution_m
        The resolution of the raster in meters.
    radius_m
        The radius of the kernel in meters.
    kernel_type
        The type of kernel to make. Either "uniform" or "gaussian".
        A uniform kernel is a circle with equal weights for all pixels inside
        the circle. A gaussian kernel is a circle with a gaussian distribution
        of weights.

    Returns
    -------
    np.ndarray
        The convolution kernel.
    """
    radius = int(radius_m // pixel_resolution_m)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]

    if kernel_type == "uniform":
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
        mask = x**2 + y**2 < radius**2
        kernel[mask] = 1 / np.sum(mask)
    elif kernel_type == "gaussian":
        kernel = np.exp(-(x**2 + y**2) / (radius**2))
        kernel = kernel / kernel.sum()
    else:
        raise NotImplementedError
    return kernel


def make_spatial_average(
    tile: rt.RasterArray,
    radius: int | float,
    kernel_type: Literal["uniform", "gaussian"] = "uniform",
) -> rt.RasterArray:
    """Compute a spatial average of a raster.

    Parameters
    ----------
    tile
        The raster to average.
    radius
        The radius of the averaging kernel in meters.
    kernel_type
        The type of kernel to use. Either "uniform" or "gaussian".

    Returns
    -------
    rt.RasterArray
        A raster with the same extent and resolution as the input raster, but
        with the values replaced by the spatial average of the input raster.
        Note that pixels within 1/2 the radius of the edge of the raster will
        have a reduced number of contributing pixels, and thus will be less
        accurate. See the documentation for scipy.signal.oaconvolve for more details.
    """
    arr = np.nan_to_num(tile.to_numpy())

    kernel = make_smoothing_convolution_kernel(tile.x_resolution, radius, kernel_type)

    out_image = oaconvolve(arr, kernel, mode="same")
    # TODO: Figure out why I did this
    out_image -= np.nanmin(out_image)
    min_value = 0.005
    out_image[out_image < min_value] = 0.0

    out_image = out_image.reshape(arr.shape)
    out_raster = rt.RasterArray(
        out_image,
        transform=tile.transform,
        crs=tile.crs,
        no_data_value=tile.no_data_value,
    )
    return out_raster
