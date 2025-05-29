import numpy as np

def improved_VMD(signal, alpha, tau, K, DC, init, tol):
    """
    Improved Variational Mode Decomposition (IVMD)
    :param signal: Input signal (1D array)
    :param alpha: Data-fidelity constraint (moderate bandwidth constraint)
    :param tau: Noise-tolerance (0 for no noise)
    :param K: Number of modes (IMFs)
    :param DC: Whether to include DC component (1 for yes, 0 for no)
    :param init: Initialization method (1: random, 0: fixed frequencies)
    :param tol: Convergence tolerance
    :return: (modes, spectrum, frequencies)
    """
    length = len(signal)
    t = np.arange(1, length + 1)
    omega = np.zeros((K,)) if init == 0 else np.sort(np.random.rand(K))
    
    # Compute FFT of the signal
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(length)
    
    # Initialize variables
    modes = np.zeros((K, length))
    spectrum = np.zeros((K, length), dtype=complex)
    u_hat = np.zeros(length, dtype=complex)
    lambda_hat = np.zeros(length, dtype=complex)
    
    iterations = 0
    previous_modes = np.copy(modes)
    
    while True:
        for k in range(K):
            residual = fft_signal - np.sum(spectrum, axis=0) + spectrum[k, :]
            spectrum[k, :] = (residual + lambda_hat / 2) / (1 + alpha * (freqs - omega[k])**2)
            modes[k, :] = np.real(np.fft.ifft(spectrum[k, :]))
            omega[k] = np.sum(freqs * np.abs(spectrum[k, :])**2) / np.sum(np.abs(spectrum[k, :])**2)
        
        # Update Lagrange multipliers
        lambda_hat += tau * (np.sum(spectrum, axis=0) - fft_signal)
        
        # Convergence check
        delta = np.linalg.norm(modes - previous_modes) / np.linalg.norm(previous_modes)
        if delta < tol or iterations > 500:
            break
        previous_modes = np.copy(modes)
        iterations += 1
    
    return modes, spectrum, omega
