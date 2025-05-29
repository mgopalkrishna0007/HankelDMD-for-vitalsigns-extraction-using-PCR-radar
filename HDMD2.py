import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel

# Step 1: Generate a 12 BPM breathing signal (0.2 Hz)
fs = 10  # sampling frequency in Hz
duration = 60  # seconds
t = np.linspace(0, duration, fs * duration)
f_breath = 0.2  # 12 bpm = 0.2 Hz
signal = np.sin(2 * np.pi * f_breath * t)

# Step 2: Construct the Hankel matrix
n = 40  # number of rows (embedding dimension)
m = 60  # number of columns (window size)
assert m + n - 1 <= len(signal), "Signal is too short for given n and m"

# Create first column and last row of the Hankel matrix
col = signal[:n]
row = signal[n-1:n+m-1]
H = hankel(col, row)

# Step 3: Print the Hankel matrix (truncated for display)
print("Hankel Matrix (showing top 5 rows and 10 columns):")
print(np.round(H[:5, :10], 3))

# Step 4: Plot the Hankel matrix
plt.figure(figsize=(10, 6))
plt.imshow(H, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Amplitude')
plt.title('Hankel Matrix of 12 BPM Breathing Signal')
plt.xlabel('Hankel Column (Time Delay)')
plt.ylabel('Hankel Row (Time Step)')
plt.grid(False)
plt.show()
