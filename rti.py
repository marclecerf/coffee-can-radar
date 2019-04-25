"""rti.py

Plot Range vs Time Intensity (RTI) for Cantenna radar

Python translation of MATLAB script provided by MIT OpenCourseware materials -

MIT IAP Radar Course 2011

Resource: Build a Small Radar System Capable of Sensing Range, Doppler, 
and Synthetic Aperture Radar Imaging 

Gregory L. Charvat
"""
import argparse
import sys
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

#NOTE: set up-ramp sweep from 2-3.2V to stay within ISM band
#change fstart and fstop below when in ISM band

# CONSTANTS
c = 3.0E8 #(m/s) speed of light

# RADAR PARAMETERS
Tp = 20.0E-3 # (s) pulse time
fstart = 2260.0E6 # (Hz) LFM start frequency for example
fstop = 2590.0E6 # (Hz) LFM stop frequency for example
#fstart = 2402E6 # (Hz) LFM start frequency for ISM band
#fstop = 2495E6 # (Hz) LFM stop frequency for ISM band
BW = fstop - fstart # (Hz) transmit bandwidth
rr = c/(2 * BW) # range resolution

def dbv(inp):
    return 20.0 * np.log10(np.absolute(inp))

def range_profiles(wav, FS, N):
    """Returns de-chirped range profiles following edge sync rising edge

    Inputs:
    - wav: 2-channel audio data
      - Assumes sync pulse is on wav channel 0
      - Assumes range profiles are on wav channel 1
    - FS: sample rate of audio data
    - N: number of samples per radar pulse

    Returns (tim, sif):
    - 'tim' is a length M vector of range profile start times
    - 'sif' is an [M x N] matrix, where the j'th row is the range profile
       at time tim[j]
    """
    #the input appears to be inverted
    trig = -1 * wav[:, 0]
    # Plot the first second of trigger
    plt.plot(trig, 'b-')
    plt.show()
    sys.exit(0)
    s = -1 * wav[:, 1]
    thresh = 0
    start = (trig > thresh)
    # Robust check for index of the rising edge of each trigger
    # Make heavy use of numpy vector operations, otherwise this
    # takes forever...
    startroll = []
    for nn in xrange(1, 12):
        startroll.append(np.roll(start, nn))
    startroll = np.array(startroll)
    mstart = np.mean(startroll, axis=0)
    check = np.logical_and(start, mstart==0)
    idxlist = np.nonzero(check)[0]
    sif = []
    tim = []
    for idx in idxlist:
        row = s[idx:idx+N]
        if row.shape[0] == N:
            sif.append(row)
            tim.append(idx * 1.0/FS)
    sif = np.array(sif)
    tim = np.array(tim)
    return tim, sif

def plots(wavpath, t0=None, tf=None):
    FS, data = wavfile.read(wavpath)
    nsamples = data.shape[0]
    nchannels = data.shape[1]
    duration = nsamples / float(FS)
    print('Reading .wav file \'%s\'' % wavpath)
    print('Sample type: %s' % str(data.dtype))
    print('Sample rate: %d Hz' % FS)
    print('Number of channels: %d' % nchannels)
    print('Total duration (s): %f' % (nsamples / float(FS)))
    print('Total number of samples: %d' % nsamples)
    if t0 is None:
        t0 = 0.
        s0 = 0
    else:
        s0 = int(t0 * FS)
    if tf is None:
        tf = duration
        sf = nsamples - 1
    else:
        sf = int(tf * FS)
    nsamples = sf - s0
    duration = nsamples / float(FS)
    data = data[s0:sf, :]
    print('Analysis start (s): %f' % t0)
    print('Analysis stop (s): %f' % tf)
    print('Analysis number of samples: %d' % nsamples)
    print('Channel 0 range: [%d, %d]' % (min(data[:, 0]), max(data[:, 0])))
    print('Channel 1 range: [%d, %d]' % (min(data[:, 1]), max(data[:, 1])))
    if data.shape[1] != 2:
        print('ERROR: .wav file must be stereo (2 channels)')
        return 1
    #return 0
    N = int(Tp * FS) # number of samples per pulse
    tim, sif = range_profiles(data, FS, N)
    # subtract the average
    ave = np.mean(sif, axis=0)
    sif = sif - ave
    max_range = rr * N/2 # max range
    zpad = 8*N/2
    def extents(f):
        delta = f[1] - f[0]
        return [f[0] - delta/2, f[-1] + delta/2]
    # RTI plot
    v = dbv(np.fft.ifft(sif, n=zpad, axis=1))
    S = v[:,0:int(v.shape[1]/2)]
    m = np.max(np.max(v))
    plt.figure()
    plt.imshow(S-m, aspect='auto', interpolation='nearest',
               extent=extents(np.linspace(0, max_range, zpad)) + extents(tim))
    plt.title('RTI without clutter rejection')
    plt.xlabel('Range (meters)')
    plt.ylabel('Time (seconds)')
    # 2-pulse canceller RTI plot
    sif2 = sif[1:sif.shape[0], :] - sif[0:sif.shape[0]-1, :]
    v2 = dbv(np.fft.ifft(sif2, n=zpad, axis=1))
    # TODO: ref MATLAB script, scale S2 by range
    S2 = v2[:,0:int(v2.shape[1]/2)]
    m2 = np.max(np.max(v2))
    plt.figure()
    plt.imshow(S2-m2, aspect='auto', interpolation='nearest',
               extent=extents(np.linspace(0, max_range, zpad)) + extents(tim))
    plt.title('RTI with 2-pulse canceller clutter rejection')
    plt.xlabel('Range (meters)')
    plt.ylabel('Time (seconds)')
    plt.show()

def parse_args():
    ap = argparse.ArgumentParser('RTI Plotter')
    ap.add_argument('wavfile', help='RADAR .wav file')
    ap.add_argument('--t0', type=float, default=None, help='Start time (s)')
    ap.add_argument('--tf', type=float, default=None, help='Stop time (s)')
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sys.exit(plots(args.wavfile, t0=args.t0, tf=args.tf))
