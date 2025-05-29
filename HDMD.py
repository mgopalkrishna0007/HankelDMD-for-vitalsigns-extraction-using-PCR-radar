import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd, eig
from sklearn.metrics import mean_squared_error

# =============================================
# Step 1: Create a Noisy Multi-Frequency Signal
# =============================================
fs = 1000  # Sampling frequency (Hz)
t = np.linspace(0, 1, fs)  # Time vector (1 second)

# Original signal components
freqs = [5, 20, 45]  # Frequencies (Hz)
amps = [1.0, 0.5, 0.3]  # Amplitudes
signal_clean = amps[0]*np.sin(2*np.pi*freqs[0]*t) + \
               amps[1]*np.cos(2*np.pi*freqs[1]*t) + \
               amps[2]*np.sin(2*np.pi*freqs[2]*t)

# Add Gaussian noise
noise = 0.2 * np.random.normal(size=len(t))
signal_noisy = signal_clean + noise

# Plot original vs noisy signal
plt.figure(figsize=(12, 4))
plt.plot(t, signal_clean, 'b', label='Clean Signal')
plt.plot(t, signal_noisy, 'r', alpha=0.6, label='Noisy Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Original vs Noisy Signal')
plt.show()

# =============================================
# Step 2: HDMD Algorithm Implementation
# =============================================
def hdmd(signal, m=None, k=None, dt=1/1000):
    N = len(signal)
    if m is None:
        m = N // 2  # Default window size
    n = N - m + 1
    
    # Hankel matrices (Eq. 1-2 in paper)
    X = hankel(signal[:m], signal[m-1:N-1])
    Y = hankel(signal[1:m+1], signal[m:N])
    
    # SVD of X (Eq. 7)
    U, S, Vh = svd(X, full_matrices=False)
    if k is None:
        k = np.sum(S > 1e-5)  # Auto-rank based on singular values
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vh_k = Vh[:k, :]
    
    # Koopman operator (Eq. 6, 9)
    A_tilde = U_k.T @ Y @ Vh_k.T @ np.linalg.inv(S_k)
    
    # Eigen decomposition (Eq. 10)
    mu, w = eig(A_tilde)
    
    # DMD modes (Eq. 11)
    Phi = Y @ Vh_k.T @ np.linalg.inv(S_k) @ w
    
    # Continuous-time eigenvalues (Eq. 19)
    s = np.log(mu) / dt
    
    # Natural frequencies (Eq. 21)
    f = np.abs(s) / (2*np.pi)
    
    return Phi, f, s, X, Y, U_k, S_k, Vh_k , mu

# Run HDMD
Phi, f_est, s_est, X, Y, U_k, S_k, Vh_k , mu = hdmd(signal_noisy, m=200, k=3)

# =============================================
# Step 3: Plot Matrices and Modes
# =============================================
# Plot Hankel matrices
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.imshow(X, aspect='auto', cmap='viridis')
plt.title('Hankel Matrix X')
plt.colorbar()

plt.subplot(122)
plt.imshow(Y, aspect='auto', cmap='viridis')
plt.title('Hankel Matrix Y')
plt.colorbar()
plt.tight_layout()
plt.show()

# Plot SVD components
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(U_k, aspect='auto', cmap='viridis')
plt.title('Left Singular Vectors (U_k)')

plt.subplot(132)
plt.stem(np.diag(S_k))
plt.title('Singular Values (S_k)')

plt.subplot(133)
plt.imshow(Vh_k, aspect='auto', cmap='viridis')
plt.title('Right Singular Vectors (Vh_k)')
plt.tight_layout()
plt.show()

# Plot DMD modes
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(t[:len(Phi[:, i])], np.real(Phi[:, i]))
    plt.title(f'DMD Mode {i+1}: Estimated Freq = {f_est[i]:.2f} Hz')
plt.tight_layout()
plt.show()

# =============================================
# Step 4: Frequency Validation
# =============================================
# Eigenvalue spectrum (Unit circle)

plt.figure(figsize=(6, 6))
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)
plt.scatter(np.real(mu), np.imag(mu), c='r', s=100)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Eigenvalues on Unit Circle')
plt.grid()
plt.axis('equal')
plt.show()

# Compare original vs estimated frequencies
print("\nFrequency Validation:")
print(f"Original Frequencies: {freqs} Hz")
print(f"HDMD Estimated Frequencies: {np.sort(np.abs(f_est))} Hz")

# =============================================
# Step 5: Reconstruction and MSE
# =============================================
# Reconstruct signal (Eq. 15)
b = np.linalg.pinv(Phi) @ signal_noisy[:len(Phi)]
t_recon = t[:len(Phi)]
reconstruction = np.zeros(len(Phi), dtype=complex)
for i in range(len(f_est)):
    reconstruction += Phi[:, i] * (b[i] * np.exp(s_est[i] * t_recon))

plt.plot(t_recon, signal_clean[:len(Phi)], 'b', label='Original Clean')
plt.plot(t_recon, np.real(reconstruction), 'r--', label='HDMD Reconstruction')


# Plot reconstruction
plt.figure(figsize=(12, 4))
plt.plot(t, signal_clean, 'b', label='Original Clean')
plt.plot(t, np.real(reconstruction), 'r--', label='HDMD Reconstruction')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Signal Reconstruction')
plt.show()

# MSE calculation
mse = mean_squared_error(signal_clean[:len(reconstruction)], np.real(reconstruction))
print(f"\nReconstruction MSE: {mse:.2e}")

# Runtime benchmark (HDMD rate)
import timeit
runtime = timeit.timeit(lambda: hdmd(signal_noisy, m=200, k=3), number=100)/100
print(f"HDMD Decomposition Rate: {1/runtime:.2f} signals/second")