import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Analysis_functions:
    @staticmethod
    def analyze_dark_frames(darks):
        """
        For each pixel compute the statistics across dark frames.
        Then summarize the statistics (mean and std distribution across the detector).
        """
        dark_mean = np.mean(darks, axis=0)
        dark_std = np.std(darks, axis=0)
        
        overall_mean = np.mean(dark_mean)
        overall_median = np.median(dark_mean)
        overall_std_mean = np.std(dark_mean)
        overall_mean_std = np.mean(dark_std)
        
        print("\nDark Frames Statistics:")
        print(f"- Mean bias level (Overall mean of pixel dark level): {overall_mean:.2f} counts")
        print(f"- Overall median of pixel dark level: {overall_median:.2f} counts")
        print(f"- Overall standard deviation of pixel dark level: {overall_std_mean:.2f} counts")
        print(f"- Mean noise level (std) (Overall mean of pixel dark level standard deviation): {overall_mean_std:.2f} counts")
        print(f"- Min/Max bias values: {np.min(dark_mean):.2f} / {np.max(dark_mean):.2f} counts")
        
        return dark_mean, dark_std, overall_mean, overall_median, overall_std_mean, overall_mean_std