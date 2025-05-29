# import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# from matplotlib import rcParams

# # Set font to Times New Roman
# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Times New Roman']
# rcParams['font.size'] = 10

# # File path
# file_path = r"C:\Users\GOPAL\guptaradardata\B2.h5"

# # Load data
# with h5py.File(file_path, "r") as f:
#     frame = f["sessions/session_0/group_0/entry_0/result/frame"]
#     real_part = np.array(frame["real"], dtype=np.float64)  # I component
#     imag_part = np.array(frame["imag"], dtype=np.float64)  # Q component

# # Combine into complex IQ data
# IQ_data = real_part + 1j * imag_part
# print("Data shape:", IQ_data.shape)  # Should be (num_frames, num_sweeps, num_range_bins)

# # Parameters for range calculation
# start_point = 0.225  # meters
# end_point = 1.525    # meters
# num_range_bins = IQ_data.shape[2]
# range_bins = np.linspace(start_point, end_point, num_range_bins)

# # Create figure
# plt.figure(figsize=(10, 6))
# plt.title('Range Profile Magnitude')
# plt.xlabel('Distance (meters)')
# plt.ylabel('Magnitude')
# plt.grid(True)

# # Set axis limits
# plt.xlim([start_point, end_point])
# plt.ylim([0, 1.1 * np.max(np.abs(IQ_data))])

# # Animation parameters
# frame_rate = 14.8  # Hz
# frame_delay = 1.0 / frame_rate  # seconds between frames

# try:
#     # Iterate through all frames
#     for frame_idx in range(IQ_data.shape[0]):
#         # Clear previous plot
#         plt.cla()
        
#         # Average across sweeps for this frame and calculate magnitude
#         magnitude = np.abs(np.mean(IQ_data[frame_idx, :, :], axis=0))
        
#         # Plot current frame
#         plt.plot(range_bins, magnitude, 'b-', linewidth=1.5)
#         plt.title(f'Range Profile Magnitude - Frame {frame_idx+1}/{IQ_data.shape[0]}')
#         plt.xlabel('Distance (meters)')
#         plt.ylabel('Magnitude')
#         plt.grid(True)
#         plt.xlim([start_point, end_point])
#         plt.ylim([0, 1.1 * np.max(np.abs(IQ_data))])
        
#         # Add time information
#         plt.text(0.02, 0.95, f'Time: {frame_idx/frame_rate:.2f}s', 
#                 transform=plt.gca().transAxes, fontsize=10)
        
#         # Pause to create animation effect
#         plt.pause(frame_delay)
        
# except KeyboardInterrupt:
#     print("\nAnimation stopped by user")

# plt.tight_layout()
# plt.show()


import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

# Set font to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 10

# File path
file_path = r"C:\Users\GOPAL\guptaradardata\B2.h5"

# Load data
with h5py.File(file_path, "r") as f:
    frame = f["sessions/session_0/group_0/entry_0/result/frame"]
    real_part = np.array(frame["real"], dtype=np.float64)  # I component
    imag_part = np.array(frame["imag"], dtype=np.float64)  # Q component

# Combine into complex IQ data
IQ_data = real_part + 1j * imag_part
print("Data shape:", IQ_data.shape)  # (num_frames, num_sweeps, num_range_bins)

# Parameters
start_point = 0.225  # meters
end_point = 1.525    # meters
num_range_bins = IQ_data.shape[2]
range_bins = np.linspace(start_point, end_point, num_range_bins)
time_axis = np.arange(IQ_data.shape[0]) / 14.8  # Time in seconds (14.8 Hz frame rate)

# Calculate magnitude for all frames (average across sweeps)
magnitude_data = np.abs(np.mean(IQ_data, axis=1))  # Shape: (num_frames, num_range_bins)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create grid for surface plot
X, Y = np.meshgrid(range_bins, time_axis)

# Plot the surface
surf = ax.plot_surface(X, Y, magnitude_data, 
                      cmap='plasma', 
                      edgecolor='none', 
                      rstride=1, 
                      cstride=1,
                      alpha=0.8)

# Add color bar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Magnitude')

# Labels and title
ax.set_xlabel('Distance (meters)', labelpad=10)
ax.set_ylabel('Time (seconds)', labelpad=10)
ax.set_zlabel('Magnitude', labelpad=10)
ax.set_title('3D Range Profile Evolution', pad=20)

# Adjust viewing angle
ax.view_init(elev=30, azim=45)

# Remove background grids for cleaner look
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.tight_layout()
plt.show()