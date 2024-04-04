import numpy as np

# Create a simple input signal
signal = np.array([0, 1, 0, 0])

# Compute the 1D DFT
dft_signal = np.fft.fft(signal)

print("Signal:", signal)
print("DFT:", dft_signal)

