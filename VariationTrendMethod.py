# this code applies variation trend method to detect the phase signal vs time 
# plot , so that the plot can be used further for analysis 

file_path = r"C:\acconeerData\breath01sparseiq.h5"

import numpy as np
import h5py
import matplotlib.pyplot as plt

# File path
# file_path = r"C:\Users\GOPAL\Downloads\amritoutdoor (2)\amritoutdoor\Sitting\amrit03sittingL.h5"

# file_path = r"C:\Users\GOPAL\guptaradardata\B2.h5"

# Load data
with h5py.File(file_path, "r") as f:
    frame = f["sessions/session_0/group_0/entry_0/result/frame"]
    real_part = np.array(frame["real"], dtype=np.float64)  # Extract real part
    imag_part = np.array(frame["imag"], dtype=np.float64)  # Extract imaginary part

# print("real_part shape:", real_part.shape)
# print("imag_part shape:", imag_part.shape)
# print("real_part ", real_part)
# print ("len of real part " , len(real_part))
# print("imag_part ", imag_part)
# print ("len of imag part " , len(imag_part))
# # Combine real and imaginary parts into complex IQ data
IQ_data = real_part + 1j * imag_part  # Shape: (1794, 32, 40)
# print("IQ_data shape:", IQ_data.shape)
# print ("IQ_data ", IQ_data)

# Transpose data to match MATLAB's order: (antennas x range bins x sweeps)
IQ_data = IQ_data.transpose(2, 1, 0)  # Shape: (40, 32, 1794)

# Original Parameters
# fs = 100  # Sweep rate (Hz)
# range_spacing = 0.5e-3  # Range spacing (m)
# D = 100  # Downsampling factor
# tau_iq = 0.5  # Time constant for low-pass filter (seconds)
# f_low = 0.1 # High-pass filter cutoff frequency (Hz)

# Tuned Parameters
fs = 36  # Sweep rate (Hz)
range_spacing = 0.5e-3  # Range spacing (m)
D = 100  # Downsampling factor
tau_iq = 0.04  # Time constant for low-pass filter (seconds)
f_low = 0.2 # High-pass filter cutoff frequency (Hz)

# Compute the magnitude of IQ data (sweeps x range bins)
magnitude_data = np.abs(IQ_data)

# Find the range bin with the highest peak magnitude (across all sweeps)
mean_magnitude = np.mean(magnitude_data, axis=2)  # Mean over sweeps
peak_range_index = np.argmax(mean_magnitude, axis=1)  # Index for each antenna

# Select the range indices based on the peak range bin
range_start_bin = max(0, peak_range_index[0] - 5)  # Adjust as needed
range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
range_indices = np.arange(range_start_bin, range_end_bin + 1)

# Downsampling
downsampled_data = IQ_data[:, range_indices[::D], :]  # Shape: (40, downsampled ranges, 1794)

# Temporal low-pass filter parameters
alpha_iq = np.exp(-2 / (tau_iq * fs))  # Low-pass filter coefficient

# Initialize filtered data
filtered_data = np.zeros_like(downsampled_data)
filtered_data[:, :, 0] = downsampled_data[:, :, 0]

# Apply temporal low-pass filter
for s in range(1, downsampled_data.shape[2]):
    filtered_data[:, :, s] = alpha_iq * filtered_data[:, :, s - 1] + \
                             (1 - alpha_iq) * downsampled_data[:, :, s]

# Phase unwrapping and high-pass filtering parameters
alpha_phi = np.exp(-2 * f_low / fs)  # High-pass filter coefficient

# Initialize phase values
phi = np.zeros(filtered_data.shape[2])  # Phase for each sweep

# Calculate phase for each sweep
for s in range(1, filtered_data.shape[2]):
    z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
    phi[s] = alpha_phi * phi[s - 1] + np.angle(z)
plt.rcParams["font.family"] = "Times New Roman"  # Set global font

# Plot the phase vs. frames with expanded x-axis
plt.figure(figsize=(12, 6))  # Increase figure width for better spacing
plt.plot(range(len(phi)), phi, linewidth=1.5 , color ="black")
plt.xticks(np.arange(0, len(phi), step=100))  # Set x-axis ticks at intervals of 100
plt.xlabel('Frame Index (sweeps)')
plt.ylabel('Phase (radians)')
plt.title('Variation Trend Method : Demodualted Signal')
# plt.grid(True)
plt.tight_layout()  # Ensure no clipping
plt.show()

