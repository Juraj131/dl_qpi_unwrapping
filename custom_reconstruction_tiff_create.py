import matplotlib
matplotlib.use('TkAgg')

import QDF
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import os
import tifffile
from skimage.restoration import unwrap_phase

SIZE = [602, 602]

def fit_and_subtract_background(image, order=2):
    """
    Fit a polynomial of given order to the image and subtract it as background.

    Parameters:
    - image: 2D numpy array, the image to process.
    - order: int, the order of the polynomial to fit.

    Returns:
    - corrected_image: 2D numpy array, the image with the fitted polynomial background subtracted.
    - poly_surface: 2D numpy array, the fitted polynomial surface.
    """
    # Generate x and y indices
    y, x = np.indices(image.shape)
    x = x.ravel()
    y = y.ravel()
    z = image.ravel()

    # Fit polynomial
    # Generate the design matrix for polynomial fitting
    # For a 2D polynomial of order 'n', we need all combinations of x^i * y^j for all i+j<=n
    coeffs = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            coeffs.append(x**i * y**j)
    A = np.vstack(coeffs).T

    # Solve for the polynomial coefficients
    poly_coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    # Evaluate polynomial
    poly_surface = np.dot(A, poly_coeffs).reshape(image.shape)

    # Subtract background
    corrected_image = image - poly_surface

    return corrected_image, poly_surface

def _reconstruct_fft(hologram, hologram_info):

    carrier = hologram_info['microscope info']['fft']['carrier']
    peak_pos = [int(carrier['x']), int(carrier['y'])]

    # print(peak_pos)

    NA_m_ref = 0.025
    NA_m = float(hologram_info['microscope info']['objectives']['NA']) / float(hologram_info['microscope info']['objectives']['magnification'])

    lam_um = float(hologram_info['microscope info']['illuminator']['wavelength']) / 1000
    

    pixel_um = float(hologram_info['image info']['pixel size']['width'])
    image_pixels = int(hologram_info['image info']['size']['width'])

    image_um = image_pixels * pixel_um

    # f_nyquist = 1 / (2 * image_um)  

    spectrum_pixel_size = 1 / pixel_um

    # RS = (2 * NA_m_ref * (2 / lam_um)) / spectrum_pixel_size
    # CS = (2 * NA_m * (2 / lam_um)) / spectrum_pixel_size
    RS = 569
    CS = 9999
    # dont know why it is not working - use fixed values


    circle_size = np.min([RS, CS])

    # RS_with_B = RS * 1.2 # 20% bigger ... it would be best to do it till 600
    RS_with_B = 600

    B = (RS_with_B - circle_size) / 2

    xv, yv = np.meshgrid(np.arange(SIZE[0]), np.arange(SIZE[1]))
    D = np.sqrt((xv - SIZE[0] / 2 ) ** 2 + (yv - SIZE[1] / 2 ) ** 2)
    r_circle = circle_size // 2
    r_circle_B = circle_size // 2 + B

    window = D <= r_circle
    ring = (D > r_circle) & (D <= r_circle_B)

    border_down = ring * (1 - (D - r_circle) / B)

    window = window + border_down

    window = np.sin(window * np.pi / 2) ** 2

    # plt.imshow(window)
    # plt.show()

    # plt.plot(window[300])
    # plt.show()


    fft_ = fftshift(fft2(hologram))
    fft_crop =  fft_[peak_pos[0] - int(SIZE[0] / 2) : peak_pos[0] + int(SIZE[0] / 2), peak_pos[1] - int(SIZE[1] / 2) : peak_pos[1] + int(SIZE[1] / 2)]


    fft_crop = fft_crop * window
    fft_crop = ifftshift(fft_crop)

    reconstructed = np.angle(ifft2(fft_crop))
    return reconstructed[1:-1, 1:-1]

def custom_reconstruction(hologram, hologram_info):
    # Reconstruct wrapped phase without background subtraction
    wrapped = _reconstruct_fft(hologram, hologram_info)
    return wrapped

if __name__ == "__main__":

    fname = "data2.qdf"
    reader = QDF.reader(fname)

    fname_bg = "background.qdf"
    reader_bg = QDF.reader(fname_bg)

    hologram_bg = reader_bg.get_image('Hologram', 0, 0, 0).reshape(2048,2048).astype(np.float64)
    hologram_info_bg = reader_bg.get_image_info('Hologram', 0, 0, 0)

    # Output directory
    out_dir = "wrapped_unwrapped_pairs"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Iterate over times and positions
    for t in range(0, 289, 20):
        for p in range(36):
            # Get hologram and its information
            hologram = reader.get_image('Hologram', t, p, 0).reshape(2048, 2048)
            hologram_info = reader.get_image_info('Hologram', t, p, 0)

            # Generate wrapped phase
            wrapped = custom_reconstruction(hologram, hologram_info)

            # Unwrap phase using traditional method
            unwrapped_bg = unwrap_phase(wrapped)

            # Subtract background from unwrapped phase
            unwrapped, poly_surface = fit_and_subtract_background(unwrapped_bg, order=3)

            # Save images in TIFF format with original data type
            wrapped_path = os.path.join(out_dir, f"wrappedbg_{t}_{p}.tiff")
            
            unwrapped_path = os.path.join(out_dir, f"unwrapped_{t}_{p}.tiff")

            tifffile.imwrite(wrapped_path, wrapped)
    
            tifffile.imwrite(unwrapped_path, unwrapped_bg)

    print("Wrapped and corrected unwrapped images have been saved.")
