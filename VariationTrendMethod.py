# ===========================================
# Variation Trend Method for Phase Extraction
# ===========================================

import numpy as np
import h5py
import matplotlib.pyplot as plt

# ---------------------------
# Load radar data
# ---------------------------
file_path = r"C:\Users\GOPAL\guptaradardata\A1.h5"

with h5py.File(file_path, "r") as f:
    frame = f["sessions/session_0/group_0/entry_0/result/frame"]
    real_part = np.array(frame["real"], dtype=np.float64)  # Real part
    imag_part = np.array(frame["imag"], dtype=np.float64)  # Imag part

# Combine into complex IQ data: shape (1794, 32, 40)
IQ_data = real_part + 1j * imag_part

# Transpose to (antennas x range bins x sweeps)
IQ_data = IQ_data.transpose(2, 1, 0)  # Shape: (40, 32, 1794)

# ---------------------------
# Parameters
# ---------------------------
num_sweeps = IQ_data.shape[2]  # number of time samples (frames)
duration = 60                  # seconds
fs = num_sweeps / duration     # sampling frequency in Hz

range_spacing = 0.5e-3         # Range spacing (m)
D = 100                        # Downsampling factor (range bins)
tau_iq = 0.04                  # Low-pass filter time constant (s)
f_low = 0.2                    # High-pass filter cutoff (Hz)

print(f"Data loaded: {num_sweeps} sweeps, duration={duration}s, fs={fs:.2f} Hz")

# ---------------------------
# Range bin selection
# ---------------------------
magnitude_data = np.abs(IQ_data)
mean_magnitude = np.mean(magnitude_data, axis=2)   # Average over time
peak_range_index = np.argmax(mean_magnitude, axis=1)  # Peak bin per antenna

range_start_bin = max(0, peak_range_index[0] - 5)
range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
range_indices = np.arange(range_start_bin, range_end_bin + 1)

print(f"Selected range bins: {range_start_bin} to {range_end_bin}")

# ---------------------------
# Temporal low-pass filtering
# ---------------------------
downsampled_data = IQ_data[:, range_indices[::D], :]  # keep sweeps intact
alpha_iq = np.exp(-2 / (tau_iq * fs))

filtered_data = np.zeros_like(downsampled_data)
filtered_data[:, :, 0] = downsampled_data[:, :, 0]

for s in range(1, downsampled_data.shape[2]):
    filtered_data[:, :, s] = alpha_iq * filtered_data[:, :, s - 1] + \
                             (1 - alpha_iq) * downsampled_data[:, :, s]

# ---------------------------
# Phase extraction (variation trend method)
# ---------------------------
alpha_phi = np.exp(-2 * f_low / fs)
phi = np.zeros(filtered_data.shape[2])

for s in range(1, filtered_data.shape[2]):
    z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
    phi[s] = alpha_phi * phi[s - 1] + np.angle(z)

# Time vector
t = np.linspace(0, duration, len(phi), endpoint=False)

# ---------------------------
# Plot results
# ---------------------------
plt.rcParams["font.family"] = "Times New Roman"

plt.figure(figsize=(12, 6))
plt.plot(t, phi, linewidth=1.5, color="black")
plt.xlabel("Time (s)")
plt.ylabel("Phase (radians)")
plt.title("Variation Trend Method: Demodulated Phase Signal")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
