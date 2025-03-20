from scipy.io import loadmat
import os

class Task_a_describe_the_data:
    @staticmethod
    def describe_data():
        """
        Load the data files and print out summary information about the data.
        
        Returns:
            darks: ndarray, shape (num_frames, height, width)
                Dark frames.
            flat: ndarray, shape (height, width)
                Flat field.
            white_spectrum: ndarray, shape (101,)
                White lamp spectrum.
            white_wavelength: ndarray, shape (101,)
                White lamp wavelength.
            calib_spectrum: ndarray, shape (86016,)
                Calibration lamp spectrum.
            calib_wavelength: ndarray, shape (86016,)
                Calibration lamp wavelength.
            calib_frame: ndarray, shape (height, width)
                Calibration frame.
        """
        # Load dimensions from each file and print out summary info.
        # Load the data files
        darks = loadmat('data/Darks_Shehata_Abdelrahmen.mat')['detector_darks']  # 10x4096x4096
        flat = loadmat('data/Flat_FrameShehata_Abdelrahmen.mat')['flatfield']  # 4096x4096

        white_lamp_data = loadmat('data/Whitelamp_Shehata_Abdelrahmen.mat')
        white_spectrum = white_lamp_data['white_spectrum'].flatten()  # 101
        white_wavelength = white_lamp_data['white_wavelength'].flatten()  # 101

        calib_lamp_data = loadmat('data/CalibrationLamp_Shehata_Abdelrahmen.mat')
        calib_spectrum = calib_lamp_data['calib_spectrum'].flatten()  # 86016
        calib_wavelength = calib_lamp_data['calib_wavelength'].flatten()  # 86016

        calib_frame = loadmat('data/Calibration_FrameAbdelrahman_Shehata.mat')['calibration_frame']  # 2048x2048
        
        # Describe the data
        num_frames, height, width = darks.shape
        detector_size = (height, width)
        num_pixels = height * width
        print("Data Description:")
        print(f"- Detector size: {detector_size}")
        print(f"- Number of pixels: {num_pixels}")
        print(f"- Dark frames: {darks.shape} (frames x height x width)")
        print(f"- Flat field: {flat.shape} (height x width)")
        print(f"- White lamp: {white_spectrum.shape} (Spectrum) and {white_wavelength.shape} (Wavelength)")
        print(f"- White lamp wavelength range: {white_wavelength.min():.2f} - {white_wavelength.max():.2f} nm")
        print(f"- Calibration lamp: {calib_spectrum.shape} (Spectrum) and {calib_wavelength.shape} (Wavelength)")
        print(f"- Calibration lamp wavelength range: {calib_wavelength.min():.2f} - {calib_wavelength.max():.2f} nm")
        print(f"- Calibration Frame: {calib_frame.shape} (height x width)")
        
        return darks, flat, white_spectrum, white_wavelength, calib_spectrum, calib_wavelength, calib_frame