from utils import *

'''
    Writen by Aryan Zoroufi, Soroush Ziaee
    At this implementation,
    We're going to calculate linear convolution and circular convolution of periodic-signal using FFT
    Notice:
            we use pycharm 2020.3 for running our program so if you use other application to run our code,
            please make all plot() comment and then to see each plot make it uncomment one by one per running.
    
'''

# Define Number of iteration
n = 3

# Define Periodic Signal
x = np.array([1, 5, 10, 4, 6, 3, 1])
h = np.array([3, 7, 6, 2, 1])

# Calculate Period of Signal x, y
N1 = x.shape[0]
N2 = h.shape[0]

# Generate Periodic signal
for i in range(n):
    x = np.append(x, x, axis=None)
for i in range(n):
    h = np.append(h, h, axis=None)

# Plot Periodic signal x, y
plot(x, 'Signal x', (15, 5))
plot(h, 'Signal y', (15, 5))

# Choose One Period
x_one_period = x[0:N1]
h_one_period = h[0:N2]

'''
    Linear Convolution using DFT
'''

# Pad the sequences h(n) and x(n) with zeros so that they are of length N = L + M - 1
N_linear = N1 + N2 - 1
x_zero_padding_linear = np.concatenate((x_one_period, np.zeros(np.abs(N_linear - N1))), axis=0)
h_zero_padding_linear = np.concatenate((h_one_period, np.zeros(np.abs(N_linear - N2))), axis=0)
plot(np.convolve(x_zero_padding_linear, h_zero_padding_linear), 'Direct Convole-method1 ', (15, 5))
# Multiply the DFTs to form the product Y (k) = H (k).X ( k )
Y_fft_linear = np.fft.fft(x_zero_padding_linear) * np.fft.fft(h_zero_padding_linear)

# Find the inverse DFT of Y (k)
y_linear = np.real(np.fft.ifft(Y_fft_linear))
plot(y_linear, 'DFT-method1', (15, 5))

'''
    Circular Convolution using DFT
'''

# Zero padding is performed to the sequence which is having lesser length, so N = max(L,M)
N_circular = max(x_one_period.shape[0], h_one_period.shape[0])
x_zero_padding_circular = np.concatenate((x_one_period, np.zeros(np.abs(N_circular - N1))), axis=0)
h_zero_padding_circular = np.concatenate((h_one_period, np.zeros(np.abs(N_circular - N2))), axis=0)
plot(np.convolve(x_zero_padding_circular, h_zero_padding_circular), 'Direct Convole-method2 ', (15, 5))
# Multiply the DFTs to form the product Y (k) = X1(k).X2(k)
Y_fft_circular = np.fft.fft(x_zero_padding_circular) * np.fft.fft(h_zero_padding_circular)

# Find the inverse DFT of Y (k)
y_circular = np.real(np.fft.ifft(Y_fft_circular))
plot(y_circular, 'DFT-method2', (15, 5))
