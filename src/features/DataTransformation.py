from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter, filtfilt
import pandas as pd

# This class removes the high frequency data (that might be considered noise) from the data.
class LowPassFilter:

    def low_pass_filter(self, data_table, col, sampling_frequency, cutoff_frequency, order=5, phase_shift=True):
        # Cutoff frequencies are expressed as the fraction of the Nyquist frequency, which is half the sampling frequency
        nyq = 0.5 * sampling_frequency
        cut = cutoff_frequency / nyq

        b, a = butter(order, cut, btype='low', output='ba', analog=False)
        if phase_shift:
            data_table[col + '_lowpass'] = filtfilt(b, a, data_table[col])
        else:
            data_table[col + '_lowpass'] = lfilter(b, a, data_table[col])
        return data_table


# Class for Principal Component Analysis. We can only apply this when we do not have missing values (i.e. NaN).
class PrincipalComponentAnalysis:

    pca = []

    def __init__(self):
        self.pca = []

    # Normalize the dataset using StandardScaler from sklearn
    def normalize_dataset(self, data_table, cols):
        scaler = StandardScaler()
        # Fit the scaler on the selected columns and transform them
        scaled_values = scaler.fit_transform(data_table[cols])
        # Convert the scaled values back into a DataFrame to maintain column structure
        data_table[cols] = pd.DataFrame(scaled_values, columns=cols, index=data_table.index)
        return data_table

    # Perform the PCA on the selected columns and return the explained variance.
    def determine_pc_explained_variance(self, data_table, cols):
        # Normalize the data first.
        data_table = self.normalize_dataset(data_table, cols)

        # Perform PCA.
        self.pca = PCA(n_components=len(cols))
        self.pca.fit(data_table[cols])
        # Return the explained variances.
        return self.pca.explained_variance_ratio_

    # Apply PCA and add new PCA columns to the dataset.
    def apply_pca(self, data_table, cols, number_comp):
        # Normalize the data first.
        data_table = self.normalize_dataset(data_table, cols)

        # Perform PCA.
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(data_table[cols])

        # Transform the old values.
        new_values = self.pca.transform(data_table[cols])

        # Add the new PCA columns.
        for comp in range(0, number_comp):
            data_table['pca_' + str(comp + 1)] = new_values[:, comp]

        return data_table
