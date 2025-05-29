import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from vmdpy import VMD  # Import VMD

# File path
file_path = r"C:\acconeerData\breath01sparseiq.h5"

# Load data
with h5py.File(file_path, "r") as f:
    frame = f["sessions/session_0/group_0/entry_0/result/frame"]
    real_part = np.array(frame["real"], dtype=np.float64)
    imag_part = np.array(frame["imag"], dtype=np.float64)

# Combine real and imaginary parts into complex IQ data
IQ_data = real_part + 1j * imag_part  # Shape: (1794, 32, 40)
IQ_data = IQ_data.transpose(2, 1, 0)  # Shape: (40, 32, 1794)

# Parameters
fs = 100  # Sweep rate (Hz)
tau_iq = 0.04  # Low-pass filter time constant
f_low = 0.2  # High-pass filter cutoff frequency
FPS = 120  # Frames per second
interval = int(1000 / FPS)  # Interval for animation (ms)
K = 4  # Number of VMD modes

# Compute magnitude and find peak range index
magnitude_data = np.abs(IQ_data)
mean_magnitude = np.mean(magnitude_data, axis=2)
peak_range_index = np.argmax(mean_magnitude, axis=1)
range_start_bin = max(0, peak_range_index[0] - 5)
range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
range_indices = np.arange(range_start_bin, range_end_bin + 1)

# Low-pass filter coefficient
alpha_iq = np.exp(-2 / (tau_iq * fs))
alpha_phi = np.exp(-2 * f_low / fs)

# Initialize filtered data and phase array
filtered_data = np.zeros_like(IQ_data[:, range_indices, :])
filtered_data[:, :, 0] = IQ_data[:, range_indices, 0]
phi = np.zeros(IQ_data.shape[2])

# **Setup real-time plots**
fig, ax = plt.subplots(K + 1, 1, figsize=(10, 8), sharex=True)

# **Phase plot**
ax[0].set_xlim(0, 2000)
ax[0].set_ylim(-6 * np.pi, 6 * np.pi)  # Keep previous phase y-axis
ax[0].set_ylabel("Phase (radians)")
ax[0].set_title("Real-Time Phase & VMD Components")
ax[0].grid(True)
line_phi, = ax[0].plot([], [], 'b-', linewidth=1.5, label="Phase Signal")
ax[0].legend()

# **VMD mode plots**
lines_vmd = [ax[i].plot([], [], linewidth=1.5, label=f"VMD Mode {i}")[0] for i in range(1, K + 1)]
for i in range(1, K + 1):
    ax[i].set_ylabel(f"VMD {i}")
    ax[i].grid(True)

ax[-1].set_xlabel("Frame Index")

# **Update function for animation**
def update(frame_idx):
    if frame_idx >= IQ_data.shape[2]:  # Stop animation when all frames are processed
        ani.event_source.stop()
        return line_phi, *lines_vmd
    
    # Apply low-pass filter
    if frame_idx > 0:
        filtered_data[:, :, frame_idx] = (
            alpha_iq * filtered_data[:, :, frame_idx - 1] +
            (1 - alpha_iq) * IQ_data[:, range_indices, frame_idx]
        )
        z = np.sum(filtered_data[:, :, frame_idx] * np.conj(filtered_data[:, :, frame_idx - 1]))
        phi[frame_idx] = alpha_phi * phi[frame_idx - 1] + np.angle(z)
    
    # **Apply VMD every 100 frames and update the same window**
    if frame_idx % 100 == 0 and frame_idx > 0:  
        u, _, _ = VMD(phi[:frame_idx], alpha=2000, tau=0, K=K, DC=0, init=1, tol=1e-6)

        # **Adaptive Y-axis scaling for each VMD mode**
        for i in range(K):
            vmd_range = np.percentile(u[i][:frame_idx], [1, 99])
            ax[i+1].set_ylim(vmd_range[0] - 0.1, vmd_range[1] + 0.1)
            lines_vmd[i].set_data(range(frame_idx), u[i][:frame_idx])

    # **Update phase plot**
    line_phi.set_data(range(frame_idx), phi[:frame_idx])

    return line_phi, *lines_vmd

# **Run animation**
ani = animation.FuncAnimation(fig, update, frames=IQ_data.shape[2], interval=interval, blit=True)
plt.show()
