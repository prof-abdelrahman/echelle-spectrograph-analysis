# Advanced Metrology - Echelle Spectrograph Analysis

## Project Overview
This project involves the analysis of data from an echelle spectrograph, including dark frames, flat frames, whitelamp spectrum, calibration lamp spectrum, and calibration frames. The analysis includes statistical processing, fitting, peak finding, and development of a wavelength solution for the detector.

## Data Description
The project uses the following data files:
- `Darks_Shehata_Abdelrahmen.mat`: 10 dark frames (10x4096x4096 double)
- `Flat_FrameShehata_Abdelrahmen.mat`: Flat frame (4096x4096 double)
- `Whitelamp_Shehata_Abdelrahmen.mat`: White lamp spectrum and wavelength (1x101 double each)
- `CalibrationLamp_Shehata_Abdelrahmen.mat`: Calibration lamp spectrum and wavelength (1x86016 double each)
- `Calibration_FrameAbdelrahman_Shehata.mat`: Calibration frame (2048x2048 double)

## Requirements
- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- Pandas
- Seaborn

## Project Structure
```
ECHELLE-SPECTROGRAPH-ANALYSIS/
├── data/                              # Data files
│   ├── Calibration_FrameAbdelrahman_Shehata.mat
│   ├── CalibrationLamp_Shehata_Abdelrahmen.mat
│   ├── Darks_Shehata_Abdelrahmen.mat
│   ├── data_description.md
│   ├── Flat_FrameShehata_Abdelrahmen.mat
│   └── Whitelamp_Shehata_Abdelrahmen.mat
├── results/                           # Generated plots and visualizations
│   ├── all_orders_overlay.png
│   ├── all_wavelength_solutions.png
│   ├── calibration_frame_image.png
│   ├── calibration_frame_peaks.png
│   ├── calibration_lamp_peaks.png
│   ├── first_10_orders_fits.png
│   ├── flat_frame_image.png
│   ├── histogram_bias_values.png
│   ├── histogram_noise_values.png
│   ├── mean_dark_frame.png
│   ├── order_identification.png
│   ├── remaining_orders_fits.png
│   ├── sensitivity_map_with_wavelength.png
│   ├── sensitivity_map.png
│   ├── sensitivity_profiles_wavelength_solution.png
│   ├── sensitivity_profiles.png
│   ├── sensitivity_vs_wavelength.png
│   ├── single_order_fit.png
│   ├── spectral_coverage.png
│   ├── std_dark_frame.png
│   └── wavelength_solution_order_[1-5].png
├── src/                               # Source code
│   ├── __pycache__/
│   ├── __init__.py
│   ├── task_a_describe_the_data.py
│   ├── task_b_dark_frame_statistics.py
│   ├── task_c_and_d_find_and_fit_the_orders.py
│   ├── task_e_determine_pixel_sensitivity.py
│   ├── task_f_find_calibration_lamp_peaks.py
│   └── task_g_develop_wavelength_solution.py
└── main.ipynb                         # Main notebook to run the analysis
```

## Tasks Performed

### Task A: Data Description
- Loads and describes the data files
- Provides information about detector size, number of pixels, etc.

### Task B: Dark Frame Statistics
- Computes statistics for each pixel across dark frames
- Visualizes mean dark frame and standard deviation
- Creates histograms of bias and noise values

### Task C & D: Order Identification and Fitting
- Identifies spectral orders in the flat frame
- Fits polynomials to the orders (order-direction)
- Fits Gaussian profiles to the order cross-sections (slit-direction)

### Task E: Pixel Sensitivity Determination
- Determines relative sensitivity of each pixel
- Implements two methods:
  1. Without wavelength solution
  2. With wavelength solution using the spectral calibration

### Task F: Calibration Lamp Peak Finding
- Identifies and fits peaks in the calibration lamp spectrum
- Provides accurate wavelength references

### Task G: Wavelength Solution Development
- Identifies spectral lines in the calibration frame
- Associates detected peaks with known wavelengths
- Develops polynomial wavelength solutions for each order
- Evaluates and visualizes the precision of the wavelength solutions

## How to Run

1. Ensure all dependencies are installed:
    ```
  pip install numpy scipy matplotlib pandas seaborn
    ```

2. Open and run the `main.ipynb` notebook:
   ```
   jupyter notebook main.ipynb
   ```

3. The notebook will execute all tasks in sequence and generate the visualizations in the results directory.

## Main Function Output
The main function returns:
- Raw data arrays
- Dark frame statistics
- Order centers and fits
- Calibration lamp peaks
- Wavelength solutions
- Sensitivity maps

## Notes
- The analysis assumes no overlap between the spectral orders
- The highest wavelength is located at the top left of the detector
- The code includes comprehensive visualizations for each analysis step
- Peak finding parameters may need adjustment based on the specific data characteristics
- The wavelength solution uses polynomial fitting to map detector positions to wavelengths
- Relative sensitivity (without wavelength solution) is normalized to the strongest pixel

## Author
Abdelrahmen Shehata