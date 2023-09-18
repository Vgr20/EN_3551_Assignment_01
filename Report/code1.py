import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Load the reuired data
mat_data = scipy.io.loadmat('signals/signal686.mat')

# Access the variable from the loaded data
data = mat_data['xn_test']
data = np.squeeze(data) 

# Visualize the data as a plot
fig,ax = plt.subplots(2,1,figsize=(20,10))
ax[0].stem(data)
ax[1].plot(data)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(0,1800)
plt.title('Signal Data from .mat File')
plt.show()

# initialize constants
sample_frequency = 128
period = 14

# Constructing several subsets of the data

s_1 = data[0:129]
s_2 = data[0:257]
s_3 = data[0:513]
s_4 = data[0:1025]
s_5 = data[0:1793]

# Finding DFT
dft_result1 = np.fft.fft(s_1)
dft_result2 = np.fft.fft(s_2)
dft_result3 = np.fft.fft(s_3)
dft_result4 = np.fft.fft(s_4)
dft_result5 = np.fft.fft(s_5)


# Initialize Frequency Array
frequency_bins = [
    np.arange((-sample_frequency/2), (sample_frequency/2), sample_frequency/len(dft_result))
    for dft_result in [dft_result1, dft_result2, dft_result3, dft_result4, dft_result5]
]
# Visualize the magnitude of the DFT result
fig,ax = plt.subplots(5,1,figsize=(20,30))
ax[0].stem(frequency_bins[0], np.abs(dft_result1))
ax[1].stem(frequency_bins[1], np.abs(dft_result2))
ax[2].stem(frequency_bins[2], np.abs(dft_result3))
ax[3].stem(frequency_bins[3], np.abs(dft_result4))
ax[4].stem(frequency_bins[4], np.abs(dft_result5))

ax[0].set_title('Dft of first 128 samples')
ax[1].set_title('Dft of first 256 samples')
ax[2].set_title('Dft of first 512 samples')
ax[3].set_title('Dft of first 1024 samples')
ax[4].set_title('Dft of first 1792 samples')

ax[0].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel('Magnitude')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Magnitude')
ax[2].set_xlabel('Frequency (Hz)')
ax[2].set_ylabel('Magnitude')
ax[3].set_xlabel('Frequency (Hz)')
ax[3].set_ylabel('Magnitude')
ax[4].set_xlabel('Frequency (Hz)')
ax[4].set_ylabel('Magnitude')

plt.show()
# DFT Averaging Method
# Initialize variables
original_array = data
L = 14

subsample_size = int(len(data)/L)
num_subsamples = len(original_array) // subsample_size

# Create Subsets with necessary samples
subsamples = [original_array[i * subsample_size:(i + 1) * subsample_size] for i in range(num_subsamples)]

# Calculation of DFT
dft_result_sub_samples = np.fft.fft(subsamples)
frequency_bins_sub_samples = np.arange((-sample_frequency/2),(sample_frequency/2),sample_frequency/subsample_size)

# Averaging DFT
average_dft = np.mean(dft_result_sub_samples, axis=0)
average_dft = np.fft.fftshift(average_dft)

# Finding the Harmonic Frequencies
sorted_array = np.argsort(np.abs(average_dft))[::-1]
top_indices = sorted_array[:8]
top_frequencies = top_indices - int(sample_frequency/2)
top_frequencies.sort()

for i in range(len(top_frequencies[4:])): 
    print("Selected harmony",i+1,":", top_frequencies[4:][i],"Hz")

# Visualize the magnitude of the DFT result
fig,ax = plt.subplots(figsize=(20,5))
ax.stem(frequency_bins_sub_samples, np.abs(average_dft))
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.title('DFT of Signal Data')
plt.show()