import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfreqz, sosfiltfilt, find_peaks
import scipy.io

# Load PPG data from CSV
csv_file_path = "c:/Users/DYlmaz/Desktop/AM based Modulation/participant_53.hdf5ppg.csv"
data = pd.read_csv(csv_file_path, skiprows=7500, nrows=10000)#400 seconds

# Step 2: Identify the data columns for the plot
y_column_index = 1  # Adjust the index based on CSV file
ppg_signal = data.iloc[:, y_column_index]

# Sample rate and desired cutoff frequency (adjust these values)
fs = 25.0  # Sample rate (Hz)
cutoff_primary = 0.12  # Primary cutoff frequency (Hz)
cutoff_secondary = 0.08  # Secondary cutoff frequency (Hz)

# Normalization
nyquist = 0.5 * fs
normal_cutoff_primary = cutoff_primary / nyquist
normal_cutoff_secondary = cutoff_secondary / nyquist

# Design a primary low-pass Butterworth filter
order = 6
sos_primary = butter(order, normal_cutoff_primary, btype='low', analog=False, output='sos')

# Apply the primary filter to the signal
primary_filtered_signal = sosfiltfilt(sos_primary, ppg_signal)

# Calculate the group delay of the primary filter
_, gd_primary = sosfreqz(sos_primary, worN=2000, fs=fs)

# Apply group delay to align the primary filtered signal
delay_samples_primary = int(np.mean(gd_primary))
aligned_primary_filtered_signal = np.roll(primary_filtered_signal, -delay_samples_primary)

# Design a secondary low-pass Butterworth filter
sos_secondary = butter(order, normal_cutoff_secondary, btype='low', analog=False, output='sos')

# Apply the secondary filter to the aligned and primary filtered signal
final_filtered_signal = sosfiltfilt(sos_secondary, aligned_primary_filtered_signal)

# Find peaks in the final filtered signal
peaks, _ = find_peaks(final_filtered_signal, height=1) 

# Calculate time intervals between peaks in milliseconds
peak_times = peaks / fs * 1000  # Convert peak indices to milliseconds

time_intervals_ms = np.diff(peak_times)  # Calculate time intervals between peaks in milliseconds


peak_times_seconds = peak_times / 1000

# Calculate respiration rate in breaths per minute (BPM) for each millisecond
respiration_rates = 60 * 1000 / time_intervals_ms

# Plot original and final filtered signals
plt.figure(figsize=(10, 6))
plt.plot(ppg_signal, label='Original Signal')
plt.plot(final_filtered_signal, label='Final Filtered Signal')
plt.plot(peaks, final_filtered_signal[peaks], 'ro', label='Detected Peaks')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.title('PPG Signal with Enhanced Noise Removal')
plt.savefig("Am_Based_Modulation.png")
plt.show()

# Plot respiration rate over time
plt.figure(figsize=(10, 6))
plt.plot(peak_times_seconds[1:], respiration_rates, label='Respiration Rate', marker='o', linestyle='-', markersize=5)  # 'o' marker style
plt.xlabel('Time (seconds)')
plt.ylabel('Respiration Rate (breaths per minute)')
plt.legend()
plt.title('Respiration Rate Over Time')
plt.savefig("Am_Based_Modulation_RR.png")
plt.show()
