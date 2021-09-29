"""rd.py

Range-Doppler processing for Cantenna radar

Python translation of MATLAB script provided by MIT OpenCourseware materials -

MIT IAP Radar Course 2011

Resource: Build a Small Radar System Capable of Sensing Range, Doppler, 
and Synthetic Aperture Radar Imaging 

Gregory L. Charvat
"""
import argparse
import logging
import sys
import numpy as np
import time
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

log = logging.getLogger(__name__)

#NOTE: set up-ramp sweep from 2-3.2V to stay within ISM band
#change fstart and fstop below when in ISM band

# CONSTANTS
c = 3.0E8 #(m/s) speed of light

# RADAR PARAMETERS
# Modulation ramp signal sent to oscillator
# (measure this with oscilloscope)
#V_ramp_min = 1.2 # (V)
#V_ramp_max = 3.7 # (V)
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
BW = fstop - fstart # (Hz) transmit bandwidth
rr = c/(2 * BW) # range resolution

def dbv(inp):
    return 20.0 * np.log10(np.absolute(inp))

def range_profile(sif, N):
    sif = np.asarray(sif)
    # subtract the average
    ave = np.mean(sif, axis=0)
    sif = sif - ave
    t0_ = time.time()
    zpad = int(8*N/2)
    v = np.absolute(np.fft.ifft(sif, n=zpad))
    tf_ = time.time()
    S = v[0:int(v.size/2)]
    max_range = rti.rr * N/2.
    R = np.linspace(0, max_range, S.size)
    return R, S

def find_rising_edges(sgn, thresh=0.):
    idxs = np.argwhere(np.logical_and(sgn[:-1] < thresh, sgn[1:] > thresh))
    return [ii[0] for ii in idxs]

def find_falling_edges(sgn, thresh=0.):
    idxs = np.argwhere(np.logical_and(sgn[:-1] > thresh, sgn[1:] < thresh))
    return [ii[0] for ii in idxs]

def range_profiles_from_wav(wav, FS, N):
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
    #trig = -1 * wav[:, 0]
    trig = wav[:, 0]
    # Plot the first second of trigger
    #plt.plot(trig, 'b-')
    #plt.show()
    #sys.exit(0)
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

def threshold_cfar(x, num_train=3, num_guard=1, rate_fa=1.0e-7):
    """Detect peaks with CFAR algorithm.

    num_train: Number of training cells.
    num_guard: Number of guard cells.
    rate_fa: False alarm rate.
    """
    train_idxs_high = np.arange(num_guard + 1, num_train + num_guard + 1)
    train_idxs_low = np.flip(train_idxs_high) * -1
    train_idxs = np.hstack((train_idxs_low, train_idxs_high))
    N = len(train_idxs)
    M = np.array([np.roll(x, ii) for ii in train_idxs])
    Pn = np.mean(M, axis=0)
    alpha = N * (rate_fa**(-1./N) - 1.)
    thresh = alpha * Pn
    return thresh

def sif_profiles(trigger, sif):
    """Returns matrix of IF signal profiles
    For each triangle ramp waveform j = 0, 1, 2, ..., J:
    row(j * 2) = ramp 'j' up
    row(j * 2 + 1) = ramp 'j' down
    """
    irise = np.asarray(find_rising_edges(trigger))
    ifall = np.asarray(find_falling_edges(trigger))
    ifall = ifall[ifall > irise[0]]
    ifall = ifall[ifall < irise[-1]]
    strig = np.sort(np.hstack((irise, ifall)))
    nprofiles = strig.size - 1
    N = int(np.min(np.diff(strig)))
    K = 10
    sif_profiles = np.zeros((nprofiles, N - K*2))
    for ii in range(nprofiles):
        sif_profiles[ii, :] = sif[(strig[ii] + K):(strig[ii] + N - K)]
    sif_profiles = sif_profiles - np.mean(sif_profiles, axis=1).reshape(-1, 1)
    #dy = 0.
    #for p in sif_profiles:
    #    plt.plot(p - np.mean(p) + dy, 'b-')
    #    dy = dy + 100
    #plt.show()
    return sif_profiles

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
    # 1. Obtain profiles from sync pulse
    sif = sif_profiles(data[:, 0], data[:, 1])
    # Back out the pulse time given the observed number of samples per pulse
    N = sif.shape[1]
    Tp = float(N) / FS
    #N = int(Tp * FS)
    max_range = rr * N/2 # max range
    # 2. FFT of profiles gives IF signal frequency peaks, which should map linearly to range
    zpad = int(8*N/2)
    fft_sif = dbv(np.fft.fft(sif, axis=1, n=zpad))
    fft_sif = fft_sif[:, 10:int(fft_sif.shape[1]/2)]
    #fft_sif = np.fft.rfft(sif, axis=1)
    #fig, ax = plt.subplots(1, 2)
    #ax[0].plot(sif[0, :])
    #ax[1].plot(fft_sif[0, :])
    #plt.show()

    fb = np.zeros((int(fft_sif.shape[0] / 2), fft_sif.shape[1]))
    fd = np.zeros((int(fft_sif.shape[0] / 2), fft_sif.shape[1]))
    for ii, _ in enumerate(fb):
        fd[ii, :] = dbv(np.absolute((fft_sif[ii * 2 + 1, :] - fft_sif[ii * 2, :]) / 2.))
        fb[ii, :] = dbv((fft_sif[ii * 2 + 1, :] + fft_sif[ii * 2, :]) / 2.)
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #ax1.imshow(dbv(fb), aspect='auto', interpolation='nearest')
    #ax1.set_title('fb (~ range)')
    #ax2.imshow(dbv(fd), aspect='auto', interpolation='nearest')
    #ax2.set_title('fd (~ doppler)')
    #plt.show()
    #sys.exit(0)
    #fb = np.diff(fb, axis=0)
    #fd = np.diff(fd, axis=0)
    
    # ANIMATION
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    lnb, = plt.plot([], [], 'b-')
    lnd, = plt.plot([], [], 'r-')

    NX = fb.shape[1]
    NT = fb.shape[0] / 2

    def init():
        ax.set_xlim(0, NX)
        b = fb[np.isfinite(fb)]
        bmax = np.max(b)
        bmin = np.min(b)
        d = fd[np.isfinite(fd)]
        dmax = np.max(d)
        dmin = np.min(d)
        vmin = np.min([bmin, bmax, dmin, dmax])
        vmax = np.max([bmin, bmax, dmin, dmax])
        ax.set_ylim(vmin, vmax)
        return lnb, lnd

    def update(frame):
        xdata = np.arange(NX)
        bdata = fb[frame * 2, :]
        ddata = fd[frame * 2 + 1, :]
        lnb.set_data(xdata, bdata)
        lnd.set_data(xdata, ddata)
        return lnb, lnd

    ani = FuncAnimation(fig, update, frames=np.arange(NT),
                        init_func=init, blit=True, interval=Tp*1000.*2)
    plt.show()

    # 3. DOPPLER PROCESSING
    #PROFILESIZE = fft_sif.shape[1]
    #NPROFILES = fft_sif.shape[0]
    #FRAMESIZE = 100  # number of profiles per doppler processing frame
    #NFRAMES = NPROFILES - FRAMESIZE
    #frames = []
    #OUTSIZE = None
    #for ii in range(NFRAMES):
    #    istart = ii
    #    iend = istart + FRAMESIZE
    #    mat = np.fft.rfft(fft_sif[istart:iend, :], axis=0)
    #    if OUTSIZE is None:
    #        OUTSIZE = mat.shape[0]
    #    elif mat.shape[0] != OUTSIZE:
    #        continue
    #    frames.append(mat)
    #frames = np.array(frames)
    #NFRAMES = frames.shape[0]

    # ANIMATION
    #fig, ax = plt.subplots()
    #im = ax.imshow(dbv(frames[0, 3:, :100]), aspect='auto', interpolation='nearest',
    #               vmin=10, vmax=100)
    ##ax.set_title('Frame 0')

    #def update(ii):
    #    im.set_data(dbv(frames[ii, 3:, :100]))
    #    #ax.set_title('Frame %d' % ii)
    #    return im,

    #ani = FuncAnimation(fig, update, frames=np.arange(NFRAMES),
    #                    blit=True, interval=Tp*1000.*2)
    #plt.show()

 
    #zpad = int(8*N/2)
    #mag_fft_sif = dbv(np.fft.ifft(sif, n=zpad))
    #mag_fft_sif = mag_fft_sif[0:int(mag_fft_sif.size/2)]
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #ax1.imshow(sif, aspect='auto', interpolation='nearest')
    #ax2.imshow(mag_fft_sif[:, :200], aspect='auto', interpolation='nearest')
    #plt.show()
    #sys.exit(0)
    #def extents(f):
    #    delta = f[1] - f[0]
    #    return [f[0] - delta/2, f[-1] + delta/2]
    ## RTI plot
    #print(sif.shape)
    #t0_ = time.time()
    #isif = np.fft.ifft(sif, axis=1, n=zpad)
    #v = dbv(isif)
    #tf_ = time.time()
    #print("ifft [%d x %d] took %f s" % (sif.shape[0], sif.shape[1], tf_ - t0_))
    ##S = v[:,0:int(v.shape[1]/2)]
    ##m = np.max(np.max(v))
    ##plt.figure()
    ##plt.imshow(S, aspect='auto', interpolation='nearest',
    ##           extent=extents(np.linspace(0, max_range, zpad)) + extents(tim))
    ##plt.title('RTI without clutter rejection')
    ##plt.xlabel('Range (meters)')
    ##plt.ylabel('Time (seconds)')
    ## 2-pulse canceller RTI plot
    #sif2 = sif[1:sif.shape[0], :] - sif[0:sif.shape[0]-1, :]
    #v2 = dbv(np.fft.ifft(sif2, axis=1))
    ## TODO: ref MATLAB script, scale S2 by range
    #S2 = v2[:,0:int(v2.shape[1]/2)]
    ##m2 = np.max(np.max(v2))
    ## CFAR
    #D2 = np.zeros(S2.shape)
    #for irow, row in enumerate(S2):
    #    thresh = threshold_cfar(row)
    #    D2[irow, :] = row > thresh
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #fig.suptitle('RTI with 2-pulse canceller clutter rejection')
    #ax1.imshow(S2, aspect='auto', interpolation='nearest',
    #           extent=extents(np.linspace(0, max_range, zpad)) + extents(tim))
    #ax1.set_xlabel('Range (meters)')
    #ax1.set_ylabel('Time (seconds)')
    #ax2.imshow(D2, aspect='auto', interpolation='nearest',
    #           extent=extents(np.linspace(0, max_range, zpad)) + extents(tim))
    #ax2.set_xlabel('Range (meters)')
    #ax2.set_ylabel('Time (seconds)')
    #plt.show()

def parse_args():
    ap = argparse.ArgumentParser('RTI Plotter')
    ap.add_argument('wavfile', help='RADAR .wav file')
    ap.add_argument('--t0', type=float, default=None, help='Start time (s)')
    ap.add_argument('--tf', type=float, default=None, help='Stop time (s)')
    return ap.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log.info('LFM ramp (GHz): [%f, %f]', fstart * 1E-9, fstop * 1E-9)
    args = parse_args()
    sys.exit(plots(args.wavfile, t0=args.t0, tf=args.tf))

