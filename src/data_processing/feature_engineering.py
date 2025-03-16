import os
from src.data_processing.preprocessing import Preprocessing

class Feature_engineering:
    @staticmethod
    def describe_data():
        # Load dimensions from each file and print out summary info.
        data_dir = os.path.join(os.getcwd(), "data")
        
        darks_file = os.path.join(data_dir, "Darks_Shehata_Abdelrahmen.mat")
        flat_file = os.path.join(data_dir, "Flat_FrameShehata_Abdelrahmen.mat")
        whitelamp_file = os.path.join(data_dir, "Whitelamp_Shehata_Abdelrahmen.mat")
        caliblamp_file = os.path.join(data_dir, "CalibrationLamp_Shehata_Abdelrahmen.mat")
        calibframe_file = os.path.join(data_dir, "Calibration_FrameAbdelrahman_Shehata.mat")
        
        detector_darks = Preprocessing.load_mat_data(darks_file, "detector_darks")
        flatfield = Preprocessing.load_mat_data(flat_file, "flatfield")
        white_data = Preprocessing.load_mat_data(whitelamp_file, "white_spectrum")
        white_wave = Preprocessing.load_mat_data(whitelamp_file, "white_wavelength")
        calib_wave = Preprocessing.load_mat_data(caliblamp_file, "calib_wavelength")
        calib_spec = Preprocessing.load_mat_data(caliblamp_file, "calib_spectrum")
        calibration_frame = Preprocessing.load_mat_data(calibframe_file, "calibration_frame")
        
        # Describe the data
        num_frames, height, width = detector_darks.shape
        detector_size = (height, width)
        num_pixels = height * width
        print("Data Description:")
        print(f"- Detector size: {detector_size}")
        print(f"- Number of frames: {num_frames}")
        print(f"- Number of pixels: {num_pixels}")
        print(f"- Dark frames: {detector_darks.shape} (frames x height x width)")
        print(f"- Flat field: {flatfield.shape} (height x width)")
        print(f"- White lamp: {white_data.shape} (Spectrum) and {white_wave.shape} (Wavelength)")
        print(f"- White lamp wavelength range: {white_wave.min():.2f} - {white_wave.max():.2f} nm")
        print(f"- Calibration lamp: {calib_spec.shape} (Spectrum) and {calib_wave.shape} (Wavelength)")
        print(f"- Calibration lamp wavelength range: {calib_wave.min():.2f} - {calib_wave.max():.2f} nm")
        print(f"- Calibration Frame: {calibration_frame.shape} (height x width)")
        
        return {
            "darks": detector_darks,
            "flat": flatfield,
            "white_spectrum": white_data,
            "white_wavelength": white_wave,
            "calib_wave": calib_wave,
            "calib_spec": calib_spec,
            "calibration_frame": calibration_frame
        }