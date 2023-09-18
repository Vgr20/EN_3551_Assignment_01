import matplotlib.pyplot as plt
import wave
import numpy as np

# Read the audio file
audio_file = wave.open('handel_audio.wav', 'r')

sample_frequency = audio_file.getframerate()
num_samples = audio_file.getnframes()
audio_data = audio_file.readframes(num_samples)
audio_array = np.frombuffer(audio_data, dtype=np.int16)

# Initailize the sample array
N = 20000
y_n = audio_array[:N]

# Creating Subsets
x_1 = y_n[0:N]
x_2 = y_n[0:N:2]
x_3 = y_n[0:N:3]
x_4 = y_n[0:N:4]

# Verifyng sets
print("shape of array : ",x_1.shape)
print("shape of array : ",x_2.shape)
print("shape of array : ",x_3.shape)
print("shape of array : ",x_4.shape)

# Visualizing the Imported signal
fig,ax = plt.subplots(figsize=(20,5))
ax.stem(y_n)
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.title('DFT of Signal Data')
plt.show()

# Calculating the DFT
dft_result_x_1 = np.fft.fft(x_1)
dft_result_x_2 = np.fft.fft(x_2)
dft_result_x_3 = np.fft.fft(x_3)
dft_result_x_4 = np.fft.fft(x_4)
dft_result_y_n = np.fft.fft(y_n)


# Defining the Interpolation Procedure
def interpolation(array,k):
    N = len(array)
    N1 = int((N+1)/2)
    N2 = int(N/2)
    if N % 2 != 0:
        array_part_1 = np.array(array[:N1-1])
        array_part_2 = np.zeros(k*N)
        array_part_3 = np.array(array[N1:N])
        interpolated_array = np.concatenate((array_part_1,array_part_2,array_part_3))
    else:
        array_part_1 = np.array(array[:N2])
        array_part_2 = np.zeros(k*N-1)
        array_part_3 = np.array(array[N2+1:N])
        term = array[N2+1] / 2
        term = np.reshape(term,1)
        interpolated_array = np.concatenate((array_part_1,term,array_part_2,term,array_part_3))
    return interpolated_array 

# Interpoating the Signals in Frequency Domain
interpolated_x_2 = interpolation(dft_result_x_2,1)
interpolated_x_3 = interpolation(dft_result_x_3,2)
interpolated_x_4 = interpolation(dft_result_x_4,3)

# Re-Constructinf the Original Signal
redefined_x_2 = np.fft.ifft(interpolated_x_2)
redefined_x_2 *= 2
redefined_x_3 = np.fft.ifft(interpolated_x_3)
redefined_x_3 *= 3
redefined_x_4 = np.fft.ifft(interpolated_x_4)
redefined_x_4 *= 4


redefined_x_n = np.fft.ifft(y_n)

# Visualizing the Signals
fig,ax = plt.subplots(4,1,figsize=(20,25))
ax[0].stem(redefined_x_2[:50])
ax[1].stem(y_n[:50])
ax[2].stem(redefined_x_2[:50])
ax[2].stem(y_n[:50],markerfmt='red')
ax[3].plot(redefined_x_2[:50] , label = "Re-constructed")
ax[3].plot(y_n[:50], label = "Original")

ax[0].set_xlabel('Samples')
ax[0].set_ylabel('Magnitude')
ax[0].set_title("Re-constructed signal")

ax[1].set_xlabel('Samples')
ax[1].set_ylabel('Magnitude')
ax[1].set_title("Originalsignal")

ax[2].set_xlabel('Samples')
ax[2].set_ylabel('Magnitude')
ax[2].set_title("Stem plots")

ax[3].set_xlabel('Samples')
ax[3].set_ylabel('Magnitude')
ax[3].set_title("Envelop Plots")
ax[3].axis('on')

plt.legend()
plt.show()

fig,ax = plt.subplots(4,1,figsize=(20,25))
ax[0].stem(redefined_x_3[:50])
ax[1].stem(y_n[:50])
ax[2].stem(redefined_x_3[:50])
ax[2].stem(y_n[:50],markerfmt='red')
ax[3].plot(redefined_x_3[:50] , label = "Re-constructed")
ax[3].plot(y_n[:50], label = "Original")


ax[0].set_xlabel('Samples')
ax[0].set_ylabel('Magnitude')
ax[0].set_title("Re-constructed signal")

ax[1].set_xlabel('Samples')
ax[1].set_ylabel('Magnitude')
ax[1].set_title("Originalsignal")

ax[2].set_xlabel('Samples')
ax[2].set_ylabel('Magnitude')
ax[2].set_title("Stem plots")

ax[3].set_xlabel('Samples')
ax[3].set_ylabel('Magnitude')
ax[3].set_title("Envelop Plots")
ax[3].axis('on')

plt.legend()
plt.show()

fig,ax = plt.subplots(4,1,figsize=(20,25))
ax[0].stem(redefined_x_4[:50])
ax[1].stem(y_n[:50])
ax[2].stem(redefined_x_4[:50])
ax[2].stem(y_n[:50],markerfmt='red')
ax[3].plot(redefined_x_4[:50] , label = "Re-constructed")
ax[3].plot(y_n[:50], label = "Original")


ax[0].set_xlabel('Samples')
ax[0].set_ylabel('Magnitude')
ax[0].set_title("Re-constructed signal")

ax[1].set_xlabel('Samples')
ax[1].set_ylabel('Magnitude')
ax[1].set_title("Originalsignal")

ax[2].set_xlabel('Samples')
ax[2].set_ylabel('Magnitude')
ax[2].set_title("Stem plots")

ax[3].set_xlabel('Samples')
ax[3].set_ylabel('Magnitude')
ax[3].set_title("Envelop Plots")
ax[3].axis('on')

plt.legend()
plt.show()

difference = y_n[:50] - redefined_x_2[:50]
norm_2 = np.linalg.norm(difference, ord=2)
print("2-norm of the difference:", norm_2)

difference = y_n[:50] - redefined_x_3[:50]
norm_2 = np.linalg.norm(difference, ord=2)
print("2-norm of the difference:", norm_2)

difference = y_n[:50] - redefined_x_4[:50]
norm_2 = np.linalg.norm(difference, ord=2)
print("2-norm of the difference:", norm_2)