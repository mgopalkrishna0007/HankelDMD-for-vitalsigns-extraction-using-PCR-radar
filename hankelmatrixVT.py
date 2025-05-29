# Hankel Matrix and DMD Analysis for Vital Signs Extraction

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd, eig

# File path (update this to your actual path if needed)
file_path = r"C:\Users\GOPAL\guptaradardata\A2.h5"

# Load data
with h5py.File(file_path, "r") as f:
    frame = f["sessions/session_0/group_0/entry_0/result/frame"]
    real_part = np.array(frame["real"], dtype=np.float64)
    imag_part = np.array(frame["imag"], dtype=np.float64)

# IQ Data processing
IQ_data = real_part + 1j * imag_part
IQ_data = IQ_data.transpose(2, 1, 0)  # Now (range_bins, antennas, sweeps)

# Parameters
fs = 14.8  # Hz
D = 100
tau_iq = 0.04
f_low = 0.2

# Magnitude for peak range bin selection
magnitude_data = np.abs(IQ_data)
mean_magnitude = np.mean(magnitude_data, axis=2)
peak_range_index = np.argmax(mean_magnitude, axis=1)

# Choose range bin around the peak
range_start_bin = max(0, peak_range_index[0] - 5)
range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
range_indices = np.arange(range_start_bin, range_end_bin + 1)

# Downsampled data (sparse)
downsampled_data = IQ_data[:, range_indices[::D], :]

# Temporal low-pass filter
alpha_iq = np.exp(-2 / (tau_iq * fs))
filtered_data = np.zeros_like(downsampled_data)
filtered_data[:, :, 0] = downsampled_data[:, :, 0]

for s in range(1, downsampled_data.shape[2]):
    filtered_data[:, :, s] = alpha_iq * filtered_data[:, :, s - 1] + \
                             (1 - alpha_iq) * downsampled_data[:, :, s]

# Phase extraction (Variation Trend)
alpha_phi = np.exp(-2 * f_low / fs)
phi = np.zeros(filtered_data.shape[2])

for s in range(1, filtered_data.shape[2]):
    z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
    phi[s] = alpha_phi * phi[s - 1] + np.angle(z)

# Plot the extracted phase signal
plt.figure(figsize=(10, 4))
plt.plot(phi, color="black")
plt.title("Extracted Phase Signal", fontsize=14)
plt.xlabel("Sweep Index")
plt.ylabel("Phase (radians)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Build Hankel matrix
N = len(phi)  # Total number of sweeps
m = 90  # number of rows
n = N - m + 1  # number of columns
H = hankel(phi[:n], phi[n-1:])

print(f"\nHankel matrix shape: {H.shape}")
print("\nSample Hankel Matrix (first 5x5 elements):")
print(np.round(H[:5, :5], 3))

# Hankel matrix visualization
plt.figure(figsize=(8, 6))
plt.imshow(H, aspect='auto', cmap='plasma')
plt.colorbar(label="Value")
plt.title("Hankel Matrix Visualization", fontsize=14)
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.tight_layout()
plt.show()

# =============================================
# Hankel-DMD Implementation
# =============================================

def hankel_dmd(X, Y, rank=None, tol=1e-10):
    """Perform Hankel-DMD with proper eigenvalue conversion"""
    U, s, Vh = svd(X, full_matrices=False)
    
    if rank is None:
        rank = np.sum(s > tol * s[0])
    
    U_r = U[:, :rank]
    s_r = s[:rank]
    Vh_r = Vh[:rank, :]
    
    A_tilde = U_r.conj().T @ Y @ Vh_r.conj().T @ np.diag(1/s_r)
    evals, evecs = eig(A_tilde)
    modes = Y @ Vh_r.conj().T @ np.diag(1/s_r) @ evecs
    
    dt = 1/fs
    omega = np.log(evals)/dt
    frequencies = np.imag(omega)/(2*np.pi)
    damping_rates = np.real(omega)
    
    return modes, evals, frequencies, damping_rates

# Prepare data matrices
X = H[:, :-1]
Y = H[:, 1:]

# Perform DMD
modes, evals, frequencies, damping_rates = hankel_dmd(X, Y)

# =============================================
# Results Analysis and Visualization
# =============================================

# Convert frequencies to BPM (for vital signs)
bpm = np.abs(frequencies) * 60  # Convert Hz to beats/min
respiration_rate = np.abs(frequencies) * 60  # breaths/min

# Sort by eigenvalue magnitude
sorted_idx = np.argsort(np.abs(evals))[::-1]
top_n = min(10, len(evals))  # Show top 10 or all if less than 10

# Print results table
print("\nTop Modes Analysis:")
print("{:<8} {:<12} {:<12} {:<12} {:<15} {:<15}".format(
    "Mode", "Freq (Hz)", "Freq (BPM)", "Damping", "|λ| (Magnitude)", "Type"))
print("-"*75)

for i in range(top_n):
    idx = sorted_idx[i]
    f_hz = frequencies[idx]
    f_bpm = np.abs(f_hz) * 60
    
    # Classify frequency
    if 0.8 <= np.abs(f_hz) <= 2.0:
        f_type = "Heart rate"
    elif 0.1 <= np.abs(f_hz) < 0.8:
        f_type = "Respiration"
    else:
        f_type = "Other"
    
    print("{:<8} {:<12.4f} {:<12.1f} {:<12.4f} {:<15.4f} {:<15}".format(
        i+1, f_hz, f_bpm, damping_rates[idx], np.abs(evals[idx]), f_type))

# 1. Eigenvalue Plot
plt.figure(figsize=(8, 8))
plt.scatter(np.real(evals), np.imag(evals), c=np.abs(evals), cmap='plasma', alpha=0.7)
plt.colorbar(label='Magnitude (|λ|)')
plt.xlabel('Real(λ)')
plt.ylabel('Imag(λ)')
plt.title('Eigenvalue Spectrum (Unit Circle)')
plt.grid(True)
plt.axhline(0, color='k', linestyle=':', alpha=0.5)
plt.axvline(0, color='k', linestyle=':', alpha=0.5)
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.5)  # Unit circle
plt.tight_layout()
plt.show()

# 2. Frequency Spectrum (Top Frequencies)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.stem(frequencies[sorted_idx], np.abs(evals[sorted_idx]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mode Magnitude (|λ|)')
plt.title('DMD Frequency Spectrum')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(bpm[sorted_idx], np.abs(evals[sorted_idx]))
plt.xlabel('Frequency (BPM)')
plt.ylabel('Mode Magnitude (|λ|)')
plt.title('Physiological Frequency Spectrum')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Stability Diagram
plt.figure(figsize=(10, 6))
sc = plt.scatter(frequencies, damping_rates, c=np.abs(evals), cmap='plasma', alpha=0.7, s=50)
plt.colorbar(sc, label='Mode Magnitude (|λ|)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Damping Rate')
plt.title('Mode Stability Diagram')
plt.axvspan(0.8, 2.0, color='green', alpha=0.1, label='Heart rate range')
plt.axvspan(0.1, 0.5, color='blue', alpha=0.1, label='Respiration range')
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Dominant Modes Visualization
top_modes = modes[:, sorted_idx[:5]]
plt.figure(figsize=(12, 8))
for i in range(5):
    plt.subplot(5, 1, i+1)
    plt.plot(np.real(top_modes[:, i]))
    plt.title(f'Mode {i+1}: {frequencies[sorted_idx[i]]:.2f} Hz ({bpm[sorted_idx[i]]:.1f} BPM)')
    plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Signal Reconstruction
t = np.arange(len(phi))
dt = 1/fs

# Project initial condition onto modes
b = np.linalg.lstsq(modes[:, sorted_idx[:5]], phi[:n], rcond=None)[0]

# Calculate time evolution
time_dynamics = np.array([b[i] * (evals[sorted_idx[i]] ** (t/dt)) for i in range(5)])

# Reconstruct signal
reconstruction = (modes[:, sorted_idx[:5]] @ time_dynamics).real[0,:len(phi)]

plt.figure(figsize=(10, 5))
plt.plot(phi, 'k-', label='Original', linewidth=1.5)
plt.plot(reconstruction, 'r--', label='DMD Reconstruction (Top 5 modes)', alpha=0.8)
plt.xlabel('Time (samples)')
plt.ylabel('Phase (radians)')
plt.title('Signal Reconstruction Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


