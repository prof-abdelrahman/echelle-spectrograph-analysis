from scipy.signal import find_peaks
import numpy as np

class PeakDetection:
    @staticmethod
    def detect_orders(flat_field):
        """
        Function to detect the order positions in the slit direction.
        
        Args:
        flat_field (ndarray): 2D array representing the flat field image.
        
        Returns:
        peaks (ndarray): Array of peak indices.
        slit_profile (ndarray): Profile of the slit direction.
        """
        # Sum along the order direction to get a profile in the slit direction
        slit_profile = np.sum(flat_field, axis=1)
        
        # Find peaks in the profile - these are the centers of the orders
        peaks, properties = find_peaks(slit_profile, height=np.max(slit_profile)*0.1, 
                                    distance=50)  # Adjust parameters as needed
        
        return peaks, slit_profile