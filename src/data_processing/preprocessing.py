from scipy.io import loadmat

class Preprocessing:
    @staticmethod
    def load_mat_data(filepath, var_name):
        """Load a variable from a MAT file."""
        data = loadmat(filepath)
        if var_name in data:
            return data[var_name]
        else:
            raise ValueError(f"Variable '{var_name}' not found in file {filepath}")