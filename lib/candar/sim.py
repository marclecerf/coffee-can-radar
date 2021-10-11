import argparse
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile

radar_model = {
    # ramp signal sent to vco, measured with oscilloscope
    'ramp_start_V': 0.4,
    'ramp_end_V': 4.48,
    'ramp_duration_s': 20e-3,
    # vco params from component datasheet
    'vco_input_min_V': 0.5,
    'vco_input_max_V': 5.0,
    'vco_output_min_hz': 2315.0e6,
    'vco_output_max_hz': 2536.0e6,
    # ADC parameters (audio signal)
    'adc_fs_hz': 44.1e3,
    'adc_nbits': 16,
    # Output signal levels (simplified model)
    # Expressed as a fraction of the ADC bit-depth
    'output_trigger_rms': 0.7,
    'output_if_rms': 0.5,
}

def vco_chirp_slope(radar_model):
    chirp_start_hz, chirp_end_hz = np.interp(
        [radar_model['ramp_start_V'], radar_model['ramp_end_V']],
        [radar_model['vco_input_min_V'], radar_model['vco_input_max_V']],
        [radar_model['vco_output_min_hz'], radar_model['vco_output_max_hz']]
    )
    return (chirp_end_hz - chirp_start_hz) / radar_model['ramp_duration_s']

def simulate_single_target(radar_model, target_range_m, duration_s, filename):
    # Simulates a single target in front of radar, writes WAV file
    #
    # It's quite difficult to accurately simulate both the RF and IF
    # signals in time domain because of the order-of-magnitude difference
    # in frequencies (for instance, it seems the low-pass butterworth filter
    # of the mixer output easily becomes ill-conditioned).
    #
    # Instead, we use a simpler model, just simulate the output of the
    # IF mixer after it has been low-passed.
    #
    # todo: add realistic noise sources
    #       incorporate low-pass filter of IF signal in the event
    #       that noise source is added ... the real mixer output is low-passed
    #       so any broadband noise should get cut as well
    #       cutoff_hz = 20.0e3
    #       Wn = cutoff_hz * 2 * np.pi
    #       butter = signal.butter(4, Wn, analog=True, output='ba', fs=None)
    #       tout, yout, _ = signal.lsim(butter, U=u, T=t)
    c_m_per_s = 3.0e8
    tau_s = 2 * target_range_m / c_m_per_s
    if_hz = vco_chirp_slope(radar_model) * tau_s
    print('if hz: ', if_hz)
    # sample IF at audio sample rate
    dt_s = 1.0 / radar_model['adc_fs_hz']
    t = np.arange(0, duration_s, dt_s)
    print('t size: ', t.shape)
    print('tmax: ', np.amax(t))
    print('dt_s: ', dt_s)
    # two channels ... trigger signal and IF signal
    # trigger is a square wave - "high" when on, and
    # "low" when off. By inspection of the real trigger,
    # there's "ringing" at its edges, meaning it's likely
    # generated via a sum of sinusoids ... so we'll do
    # that here too for realism.
    # Ref: https://mathworld.wolfram.com/FourierSeriesSquareWave.html
    def unit_square_wave(t, nterms=30):
        # Square wave with amplitude=1, period=1s
        L = 1.0/2
        terms = np.zeros((nterms, t.size))
        for ii in range(nterms):
            n = 2 * ii + 1
            terms[ii] = (1 / n) * np.sin(n * np.pi * t / L)
        sum = np.sum(terms, axis=0)
        return (4 / np.pi) * sum
    # trigger period = 2 * [ramp duration], because we're
    # generating a triangle wave
    T = radar_model['ramp_duration_s']
    unit_out_trig = unit_square_wave(t / T)
    # IF signal.
    unit_out_if = np.sin(2 * np.pi * if_hz * t)
    # Scale outputs
    def rms(x):
        return np.sqrt(np.sum(x**2) / x.size)
    out_trig_scaled = radar_model['output_trigger_rms'] / rms(unit_out_trig) * unit_out_trig
    out_if_scaled = radar_model['output_if_rms'] / rms(unit_out_if) * unit_out_if
    adc_min = -2**(radar_model['adc_nbits'] - 1)
    adc_max = 2**(radar_model['adc_nbits'] - 1) - 1
    out_trig_channel = (out_trig_scaled * adc_max).astype(np.int16)
    out_if_channel = (out_if_scaled * adc_max).astype(np.int16)
    # Write wavfile
    rate = int(radar_model['adc_fs_hz'])
    data = np.hstack((
        out_trig_channel.reshape(-1, 1),
        out_if_channel.reshape(-1, 1)
    ))
    print(data.shape)
    fig, ax = plt.subplots(2, sharex=True)
    #xlim = [0, 1000 * 20e-3]
    xlim = [0, 20e-3]
    ylim = [adc_min, adc_max]
    ax[0].plot(t, data[:, 0])
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[1].plot(t, data[:, 1])
    ax[1].set_ylim(ylim)
    plt.show()
    
    wavfile.write(filename, rate, data)
    return 0

def parse_args():
    ap = argparse.ArgumentParser('RADAR simulator (writes WAV file)')
    ap.add_argument('filename', help='output simulated .wav file')
    ap.add_argument('range', type=float, help='range to target (meters)')
    ap.add_argument('duration', type=float, help='duration of output (seconds)')
    return ap.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    target_range_m = args.range
    duration_s = args.duration
    filename = args.filename
    sys.exit(
        simulate_single_target(
            radar_model, target_range_m, duration_s, filename
        )
    )
