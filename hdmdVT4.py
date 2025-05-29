# Hankel Matrix and DMD Analysis for Vital Signs Extraction

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd, eig

# File path (update this to your actual path if needed)
file_path = r"c:\Users\User\OneDrive\Desktop\gopalpythonfiles\A3.h5"

# Load data
with h5py.File(file_path, "r") as f:
    frame = f["sessions/session_0/group_0/entry_0/result/frame"]
    real_part = np.array(frame["real"], dtype=np.float64)
    imag_part = np.array(frame["imag"], dtype=np.float64)

# IQ Data processing
IQ_data = real_part + 1j * imag_part
IQ_data = IQ_data.transpose(2, 1, 0)  # Now (range_bins, antennas, sweeps)

# Parameters
fs = 14.8 # Hz
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
m = 150  # number of rows
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

# =============================================
# Signal Reconstruction with Diagonal Averaging
# =============================================

# 1. Time vector based on number of Hankel columns
timesteps = np.arange(H.shape[1])  # n = N - m + 1

# 2. Compute initial amplitudes (project initial snapshot onto modes)
b = np.linalg.lstsq(modes, X[:, 0], rcond=None)[0]  # shape (r,)

# 3. Compute dynamics over time using eigenvalues
dt = 1/fs
omega = np.log(evals) / dt
time_dynamics = np.array([
    b[i] * np.exp(omega[i] * timesteps)
    for i in range(len(b))
])

# 4. Reconstruct Hankel-approximated signal
X_dmd = (modes @ time_dynamics).real  # shape (m, n-1)

# 5. Reconstruct original 1D signal from Hankel approximation using diagonal averaging
def diagonal_averaging(H):
    """Convert Hankel matrix back to 1D signal"""
    m, n = H.shape
    L = m + n - 1
    result = np.zeros(L)
    count = np.zeros(L)
    for i in range(m):
        for j in range(n):
            result[i+j] += H[i, j]
            count[i+j] += 1
    return result / count

phi_reconstructed = diagonal_averaging(X_dmd)

# 6. Plot comparison
plt.figure(figsize=(12, 5))
plt.plot(phi, label='Original Signal', color='black')
plt.plot(phi_reconstructed[:len(phi)], '--', label='Reconstructed (DMD)', color='red')
plt.title('Original vs DMD Reconstructed Signal')
plt.xlabel('Sweep Index')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =============================================
# NEW: Properly Scaled Mode Comparison Plots
# =============================================

# Normalize the modes for better visualization
def normalize_mode(mode, original):
    """Scale mode to match original signal range"""
    mode_real = np.real(mode)
    scale_factor = np.std(original[:len(mode_real)]) / np.std(mode_real)
    return mode_real * scale_factor

# Create time indices for plotting
time_idx = np.arange(len(phi))

plt.figure(figsize=(12, 8))

# Plot Mode 1 comparison
plt.subplot(2, 1, 1)
plt.plot(time_idx, phi, 'k-', label='Original Signal', linewidth=1.5)
mode1_scaled = normalize_mode(modes[:, sorted_idx[0]], phi)
plt.plot(time_idx[:len(mode1_scaled)], mode1_scaled, 'b--', 
         label=f'Mode 1 ({frequencies[sorted_idx[0]]:.2f} Hz, {bpm[sorted_idx[0]]:.1f} BPM)', linewidth=1.5)
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.title('Original Signal vs Mode 1 (Scaled)')
plt.legend()
plt.grid(True)

# Plot Mode 2 comparison
plt.subplot(2, 1, 2)
plt.plot(time_idx, phi, 'k-', label='Original Signal', linewidth=1.5)
mode2_scaled = normalize_mode(modes[:, sorted_idx[1]], phi)
plt.plot(time_idx[:len(mode2_scaled)], mode2_scaled, 'r:', 
         label=f'Mode 2 ({frequencies[sorted_idx[1]]:.2f} Hz, {bpm[sorted_idx[1]]:.1f} BPM)', linewidth=2.0)
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.title('Original Signal vs Mode 2 (Scaled)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# =============================================
# CORRECTED Signal Reconstruction with Proper Scaling
# =============================================

# 1. Select only the most significant modes (avoid noise)
# Choose modes with eigenvalues close to unit circle and reasonable frequencies
significant_mask = (np.abs(evals) > 0.1) & (np.abs(frequencies) < 5.0)  # Filter unrealistic frequencies
significant_indices = np.where(significant_mask)[0]

# Sort by eigenvalue magnitude and take top modes
sorted_significant = significant_indices[np.argsort(np.abs(evals[significant_indices]))[::-1]]
n_modes = min(10, len(sorted_significant))  # Use top 10 significant modes
selected_modes = sorted_significant[:n_modes]

print(f"Using {n_modes} significant modes for reconstruction")

# 2. Improved initial condition estimation
# Use pseudoinverse for better conditioning
modes_selected = modes[:, selected_modes]
b_improved = np.linalg.pinv(modes_selected) @ X[:, 0]  # More stable than lstsq

# 3. Time evolution with selected modes only
dt = 1/fs
omega_selected = np.log(evals[selected_modes]) / dt
time_steps = np.arange(X.shape[1])  # Use actual time steps

# Create time dynamics matrix
time_dynamics_improved = np.zeros((len(selected_modes), len(time_steps)), dtype=complex)
for i, idx in enumerate(selected_modes):
    time_dynamics_improved[i, :] = b_improved[i] * np.exp(omega_selected[i] * time_steps * dt)

# 4. Reconstruct with selected modes
X_dmd_improved = np.real(modes_selected @ time_dynamics_improved)

# 5. Improved diagonal averaging with proper indexing
def diagonal_averaging_improved(H_matrix):
    """Convert Hankel matrix back to 1D signal with proper handling"""
    m, n = H_matrix.shape
    L = m + n - 1
    result = np.zeros(L)
    
    for k in range(L):
        # For each diagonal, collect all elements
        elements = []
        for i in range(max(0, k-n+1), min(m, k+1)):
            j = k - i
            if 0 <= j < n:
                elements.append(H_matrix[i, j])
        
        # Average elements on this diagonal
        if elements:
            result[k] = np.mean(elements)
    
    return result

# Apply improved diagonal averaging
phi_reconstructed_improved = diagonal_averaging_improved(X_dmd_improved)

# 6. Ensure same length and proper scaling
min_length = min(len(phi), len(phi_reconstructed_improved))
phi_orig_trimmed = phi[:min_length]
phi_recon_trimmed = phi_reconstructed_improved[:min_length]

# Apply scaling to match original signal statistics
mean_orig = np.mean(phi_orig_trimmed)
std_orig = np.std(phi_orig_trimmed)
mean_recon = np.mean(phi_recon_trimmed)
std_recon = np.std(phi_recon_trimmed)

# Scale and shift reconstructed signal
if std_recon > 1e-10:  # Avoid division by zero
    phi_recon_scaled = (phi_recon_trimmed - mean_recon) * (std_orig / std_recon) + mean_orig
else:
    phi_recon_scaled = phi_recon_trimmed + (mean_orig - mean_recon)

# 7. Alternative: Direct mode summation approach (often more stable)
def direct_reconstruction(phi_signal, modes_matrix, evals_array, fs_sampling, n_reconstruct_modes=5):
    """Direct reconstruction using dominant modes"""
    # Select top modes by eigenvalue magnitude
    top_indices = np.argsort(np.abs(evals_array))[::-1][:n_reconstruct_modes]
    
    # Get corresponding frequencies
    dt = 1/fs_sampling
    omega = np.log(evals_array[top_indices]) / dt
    frequencies_top = np.imag(omega) / (2 * np.pi)
    
    # Create time vector
    t = np.arange(len(phi_signal)) * dt
    
    # Fit amplitudes and phases using least squares
    # Create basis functions matrix
    basis_matrix = np.zeros((len(t), 2 * len(top_indices)))
    for i, freq in enumerate(frequencies_top):
        basis_matrix[:, 2*i] = np.cos(2 * np.pi * freq * t)      # Cosine component
        basis_matrix[:, 2*i+1] = np.sin(2 * np.pi * freq * t)    # Sine component
    
    # Solve for coefficients
    coefficients = np.linalg.lstsq(basis_matrix, phi_signal, rcond=None)[0]
    
    # Reconstruct signal
    reconstructed = basis_matrix @ coefficients
    
    return reconstructed, frequencies_top

# Apply direct reconstruction
phi_direct_recon, freqs_used = direct_reconstruction(phi, modes, evals, fs, n_reconstruct_modes=8)

# 8. Enhanced visualization with multiple methods
plt.figure(figsize=(15, 10))

# Plot 1: Original vs Improved DMD
plt.subplot(3, 1, 1)
time_vector = np.arange(min_length)
plt.plot(time_vector, phi_orig_trimmed, 'k-', label='Original Signal', linewidth=2)
plt.plot(time_vector, phi_recon_scaled, 'r--', label='DMD Reconstructed (Improved)', linewidth=2, alpha=0.8)
plt.title('Original vs Improved DMD Reconstructed Signal')
plt.xlabel('Sample Index')
plt.ylabel('Phase (radians)')
plt.legend()
plt.grid(True, alpha=0.3)

# Calculate and display reconstruction error
rmse_improved = np.sqrt(np.mean((phi_orig_trimmed - phi_recon_scaled)**2))
correlation_improved = np.corrcoef(phi_orig_trimmed, phi_recon_scaled)[0, 1]
plt.text(0.02, 0.98, f'RMSE: {rmse_improved:.4f}\nCorrelation: {correlation_improved:.4f}', 
         transform=plt.gca().transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 2: Direct reconstruction comparison
plt.subplot(3, 1, 2)
plt.plot(time_vector, phi_orig_trimmed, 'k-', label='Original Signal', linewidth=2)
plt.plot(time_vector, phi_direct_recon[:min_length], 'b:', label='Direct Reconstruction', linewidth=2)
plt.title('Original vs Direct Frequency-Based Reconstruction')
plt.xlabel('Sample Index')
plt.ylabel('Phase (radians)')
plt.legend()
plt.grid(True, alpha=0.3)

# Calculate metrics for direct method
rmse_direct = np.sqrt(np.mean((phi_orig_trimmed - phi_direct_recon[:min_length])**2))
correlation_direct = np.corrcoef(phi_orig_trimmed, phi_direct_recon[:min_length])[0, 1]
plt.text(0.02, 0.98, f'RMSE: {rmse_direct:.4f}\nCorrelation: {correlation_direct:.4f}', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Plot 3: Residual analysis
plt.subplot(3, 1, 3)
residual_improved = phi_orig_trimmed - phi_recon_scaled
residual_direct = phi_orig_trimmed - phi_direct_recon[:min_length]
plt.plot(time_vector, residual_improved, 'r-', label='DMD Residual', alpha=0.7)
plt.plot(time_vector, residual_direct, 'b-', label='Direct Residual', alpha=0.7)
plt.title('Reconstruction Residuals')
plt.xlabel('Sample Index')
plt.ylabel('Residual Error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print reconstruction quality metrics
print("\nReconstruction Quality Metrics:")
print("-" * 40)
print(f"Improved DMD Method:")
print(f"  RMSE: {rmse_improved:.6f}")
print(f"  Correlation: {correlation_improved:.6f}")
print(f"  Used {n_modes} modes")
print(f"\nDirect Frequency Method:")
print(f"  RMSE: {rmse_direct:.6f}")
print(f"  Correlation: {correlation_direct:.6f}")
print(f"  Used frequencies: {freqs_used}")

# Display dominant frequencies used in reconstruction
print(f"\nDominant frequencies used in DMD reconstruction:")
for i, idx in enumerate(selected_modes[:5]):
    freq_hz = frequencies[idx]
    freq_bpm = np.abs(freq_hz) * 60
    print(f"  Mode {i+1}: {freq_hz:.3f} Hz ({freq_bpm:.1f} BPM)")