import argparse
import threading
import time
import sys

import numpy as np
import pyqtgraph as pg
import pyaudio

import interface
import rti

Tp = 20.0E-3 # (s) pulse time
K = 20
c = 3e8

def dbv(inp):
    y = 20.0 * np.log10(np.absolute(inp))
    y[np.isneginf(y)] = -1000.
    y[np.isposinf(y)] = +1000.
    return y

def range_profile(sif, N):
    sif = np.asarray(sif)
    # subtract the average
    ave = np.mean(sif, axis=0)
    sif = sif - ave
    t0_ = time.time()
    zpad = int(8*N/2)
    v = np.absolute(np.fft.ifft(sif, n=zpad))
    #v = dbv(v)
    tf_ = time.time()
    S = v[0:int(v.size/2)]
    return S

def find_rising_edges(sgn, thresh=0.):
    idxs = np.argwhere(np.logical_and(sgn[:-1] < thresh, sgn[1:] > thresh))
    return [ii[0] for ii in idxs]

def find_falling_edges(sgn, thresh=0.):
    idxs = np.argwhere(np.logical_and(sgn[:-1] > thresh, sgn[1:] < thresh))
    return [ii[0] for ii in idxs]

class FpsEstimator(object):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.fps = 0.
        self.tprev = time.time()
        self.tprint = time.time()
        self.dprint = 1.0
    def update(self, nframes):
        tnow = time.time()
        dt = tnow - self.tprev
        new_fps = nframes / dt
        self.fps = (1. - self.alpha) * self.fps + self.alpha * new_fps
        self.tprev = tnow
        if tnow - self.tprint > self.dprint:
            print('FPS: %4.1f kHz' % (self.fps / 1000.))
            self.tprint = tnow

class RadarPlotter(object):
    def __init__(self, iface, size=(600,350)):
        # Data stuff
        self.lock = threading.RLock()
        self.buf0 = np.array([])
        self.buf1 = np.array([])
        self.y0 = [0]
        self.y1 = [0]
        self.x0 = [0]
        self.x1 = [0]
        self.cfar = [0]
        self.range0 = np.array([0., 0.])
        self.range1 = np.array([0., 0.])
        # PyQtGraph stuff
        self.app = pg.Qt.QtGui.QApplication([])
        self.view = pg.GraphicsView()
        l = pg.GraphicsLayout(border=(100, 100, 100))
        self.view.setCentralItem(l)
        self.view.show()
        self.view.setWindowTitle('Live Audio')
        self.view.resize(*size)
        self.plt = []
        self.curve = []
        self.plt.append(l.addPlot(title="Channel 1"))
        l.nextRow()
        self.plt.append(l.addPlot(title="Channel 2"))
        self.plt[0].showGrid(x=True, y=True)
        self.plt[1].showGrid(x=True, y=True)
        self.curve.append(self.plt[0].plot([], [], pen=(255,0,0)))
        self.curve.append(self.plt[1].plot([], [], pen=(255,0,0)))
        self.curve.append(self.plt[1].plot([], [], pen=(0,255,0)))
        self.plt[1].setXRange(0, 100, padding=None)
        self.plt[1].setYRange(0, 40, padding=None)
        # Audio stuff
        self.fps = FpsEstimator()
        self.iface = iface(self.stream_callback)
        self.chunksize = self.iface.chunksize
        self.FS = self.iface.framerate
        self.N = int(Tp * self.FS)
        # Timer
        self.timer = pg.Qt.QtCore.QTimer()
        self.timer.setSingleShot(False)
        ival = int(float(self.chunksize) / self.FS * 1000.)
        self.timer.setInterval(ival)
        self.timer.timeout.connect(self.updateplot)
        # Start things
        self.timer.start()
        self.iface.start()

    def stream_callback(self,
                        in_data,      # recorded data if input=True; else None
                        frame_count,  # number of frames
                        time_info,    # dictionary
                        status_flags): # PaCallbackFlags
        self.fps.update(frame_count)
        with self.lock:
            out = np.frombuffer(in_data, dtype=np.int16)
            outmat = np.reshape(out, (1024, 2))
            self.buf0 = np.hstack((self.buf0, outmat[:, 0]))
            self.buf1 = np.hstack((self.buf1, outmat[:, 1]))
            irising = find_rising_edges(self.buf0)
            ifalling = find_falling_edges(self.buf0)
            if len(irising) > 0 and len(ifalling) > 0:
                istart = irising[0]
                iend_idx = np.searchsorted(ifalling, istart)
                if iend_idx < len(ifalling):
                    iend = ifalling[iend_idx]
                    # Grab radar range profile + CFAR threshold
                    self.y1 = self.buf1[istart:iend]
                    self.y1 = range_profile(self.y1, self.N)
                    max_range = rti.rr * self.N/2.
                    R = np.linspace(0, max_range, self.y1.size)
                    self.x1 = R
                    _, self.cfar = rti.threshold_cfar(self.y1,
                            num_train=20, num_guard=2, rate_fa=0.2)
                    # Grab the trigger signal + a couple extra frames
                    # for better visual on the rising/falling edge
                    istart = np.max([0, istart - 10])
                    iend = np.min([iend + 10, len(self.buf0)])
                    self.y0 = self.buf0[istart:iend]
                    self.x0 = np.arange(self.y0.shape[0])/float(self.FS)
                    # Clear the data buffers
                    self.buf0 = []
                    self.buf1 = []
        return None, pyaudio.paContinue

    def updateplot(self):
        with self.lock:
            self.curve[0].setData(self.x0, self.y0)
            self.curve[1].setData(self.x1, self.y1)
            self.curve[2].setData(self.x1, self.cfar)
            self.range0[0] = np.min([self.range0[0], np.min(self.y0)])
            self.range0[1] = np.max([self.range0[0], np.max(self.y0)])
            self.plt[0].setYRange(*self.range0, padding=None)

    def run(self):
        self.app.exec_()

def parse_args():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers()
    sp_pb = sp.add_parser('playback')
    sp_pb.add_argument('file', help='input playback .wav file')
    sp_pb.set_defaults(iface=make_playback)
    IDX = 3
    sp_dev = sp.add_parser('pyaudio')
    sp_dev.add_argument('-d', '--dev',
                        help='PyAudio device index (default: %d)' % IDX,
                        default=IDX)
    sp_dev.set_defaults(iface=make_device)
    return p.parse_args()

def make_playback(args):
    return lambda cb: interface.WaveFileReader(args.file, cb)

def make_device(args):
    return lambda cb: interface.PyAudioReader(args.dev, cb)

def main():
    args = parse_args()
    m = RadarPlotter(args.iface(args))
    return m.run()

