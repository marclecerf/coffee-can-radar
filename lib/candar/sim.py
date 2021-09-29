import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

#import wave
#w = wave.open('sound.wav', 'wb')
#w.setnchannels(2)

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    print(normal_cutoff)
    return signal.butter(order, normal_cutoff, btype='low',
                         analog=True, output='zpk')

# 
# def butter_lowpass_filter(data, cutoff, fs, order=4):
#     print('pre butter cons')
#     b, a = butter_lowpass(cutoff, fs, order)
#     print('post butter cons')
#     y = signal.lfilter(b, a, data)
#     return y

# CONSTANTS
c = 3.0E8 #(m/s) speed of light

# RADAR PARAMETERS
# Modulation ramp signal sent to oscillator
# (measure this with oscilloscope)
V_ramp_min = 0.4 # (V)
V_ramp_max = 4.48 # (V)
Tp = 20.0E-3 # (s) pulse time
def osc_hz_out(v_in):
    # Oscillator min/max input voltage and
    # min/max output frequency (read these
    # off of oscillator data sheet)
    V_osc_min = 0.5 # (V)
    V_osc_max = 5.0 # (V)
    f_osc_min = 2315.0e6 # (Hz)
    f_osc_max = 2536.0e6 # (Hz)
    return np.interp(v_in, [V_osc_min, V_osc_max], [f_osc_min, f_osc_max])
fstart = osc_hz_out(V_ramp_min)  # (Hz) LFM start
fstop = osc_hz_out(V_ramp_max)  # (Hz) LFM stop
dfdt = (fstop - fstart) / Tp  # Chirp slope
# BW = fstop - fstart # (Hz) transmit bandwidth
# rr = c/(2 * BW) # range resolution


# Chirp RF signal
def chirp(t):
    return np.sin(2 * np.pi * ((dfdt/2)*(t**2) + fstart * t))

# Mixer output signal (noiseless)
def mixer(t, range_m):
    tau = 2 * range_m / c # Time delay due to round-trip time of signal
    return chirp(t) * chirp(t - tau)

# Simulate low-pass filter of mixer output
range_m = 100.0  # target distance (m)
T = 20e-3  # one chirp window

fs = 48000
t = np.arange(0, T, 1.0 / fs)
print('Sample rate (Hz): ', fs)
print('Number of samples: ', t.size)

u = mixer(t, range_m)

cutoff_hz = 20.0e3
Wn = cutoff_hz * 2 * np.pi
butter = signal.butter(4, Wn, analog=True, output='ba', fs=None)
tout, yout, _ = signal.lsim(butter, U=u, T=t)

# Get PSD of output signal
fo, po = signal.welch(
    yout, fs=fs, window='hann',
    nperseg=2048, noverlap=1024, nfft=None,
    detrend=False, return_onesided=True,
    scaling='density', average='mean'
)

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, u, 'r-')
ax[1].plot(tout, yout, 'b-')
ax[2].semilogy(fo, po, 'b-')

plt.show()

# npts = int(Tp)
# 
# npts = int(fstop * 2 * Tp)
# t = np.linspace(0, Tp, npts)
# 
# plt.plot(t, y, '-')
# plt.show()

# sample_rate_hz = 100.0e9
# # duration_s = 5.0e-8
# duration_s = 10.0e-5
# frequency_hz = 2.0e9
# cutoff_frequency_hz = 20.0e3
# nt = int(sample_rate_hz * duration_s)
# 
# t = np.linspace(0., duration_s, nt)
# 
# 
# y_t = a_t * np.sin(2. * np.pi * frequency_hz * t)
# y_r = a_r * np.sin(2. * np.pi * frequency_hz * (t - tau))
# # y_r += 0.5 * np.random.randn(*y_r.shape)
# y_m = y_t * y_r
# # print('pre-lowpass')
# y_m_filt = butter_lowpass_filter(y_m, cutoff_frequency_hz, sample_rate_hz)
# # print('post-lowpass')
# #y_m_filt = y_m
# # f_m = 20 * np.log10(np.absolute(np.fft.fft(y_m_filt)))


# Plot the frequency response.
# Get the filter coefficients so we can check its frequency response.
# cutoff_frequency_hz = 20.0e3
# sample_rate_hz = 20.0e9
# print(cutoff_frequency_hz)
# print(sample_rate_hz)
# b, a = butter_lowpass(cutoff_frequency_hz, sample_rate_hz, 4)
# w, h = signal.freqz(b, a, worN=1000000)
# plt.subplot(2, 1, 1)
# plt.semilogx(0.5 * sample_rate_hz * w / np.pi, np.abs(h), 'b')
# plt.semilogx(cutoff_frequency_hz, 0.5 * np.sqrt(2), 'ko')
# plt.axvline(cutoff_frequency_hz, color='k')
# plt.xlim(100, cutoff_frequency_hz * 10)
# plt.title("Lowpass Filter Frequency Response")
# plt.xlabel('Frequency [Hz]')
# plt.grid()
# plt.show()

# fig, ax = plt.subplots(4, 1)
# ax[0].plot(t, y_t)
# ax[0].set_title('transmitted signal')
# ax[1].plot(t, y_r)
# ax[1].set_title('received signal')
# ax[2].plot(t, y_m_filt)
# ax[2].set_title('IF mixer output (low-passed)')
# # ax[3].plot(f_m)
# plt.show()

