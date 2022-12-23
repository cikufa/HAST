import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('seaborn-poster')

def FFT(x):
    N = len(x)
    
    if N == 1:
      
        return x
    else:
        # print("xxxxx1111111", x.shape)
        X_even = FFT(x[::2])
        # print("xxxxx22222", x.shape)
        X_odd = FFT(x[1::2])
        
        factor = \
          np.exp(-2j*np.pi*np.arange(N)/ N)


        # print("ffffffffff", factor.shape)
        
        X = np.concatenate(\
            [X_even+factor[:int(N/2)]*X_odd,
             X_even+factor[int(N/2):]*X_odd])
       
        return X


data= pd.read_csv("data.csv") 
data = data.values
x=data[:,2:3]
x=x.reshape(x.shape[0])

# sampling rate
sr = x.shape[0]

# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)

# plt.figure(figsize = (8, 6))
# plt.plot(t[0:200], x[0:200], 'r')
# plt.ylabel('Amplitude')
# plt.show()


X=FFT(x)
# calculate the frequency
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 

plt.figure(figsize = (12, 6))
plt.subplot(121)
plt.stem(freq, abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')

# Get the one-sided specturm
n_oneside = N//2
# get the one side frequency
f_oneside = freq[:n_oneside]

# normalize the amplitude
X_oneside =X[:n_oneside]/n_oneside

plt.subplot(122)
plt.stem(f_oneside, abs(X_oneside), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('Normalized FFT Amplitude |X(freq)|')
plt.tight_layout()
plt.show()
