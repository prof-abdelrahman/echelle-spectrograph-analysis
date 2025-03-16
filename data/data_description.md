# Data Description

## My Data

As announced, I am getting data from an echelle spectrograph. Specifically, I get the following frames:

1. **Dark frames (10)**: This is the detector without any light incident. To read off the bias and make noise statistics for each pixel.
2. **Whitelamp (1)**: This is the spectrum of a continuous light source. It is given in Intensity/nm.
3. **Flat Frame (1)**: The light of the white lamp is sent through the spectrograph. Different orders can be seen on the detector.
4. **Calibration Lamp (1)**: This is a calibration lamp with distinctive (and apparently very regular) spectral features. Also given in Intensity/nm.
5. **Calibration Frame (1)**: The light of the calibration lamp is sent through the spectrograph. One can see the spectral features distributed across the orders.