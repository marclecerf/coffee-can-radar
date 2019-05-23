import argparse
import logging
import threading
import time
import sys

import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt
import scipy.cluster.vq
import sklearn.cluster

import candar.interface as interface
import candar.rti as rti

log = logging.getLogger(__name__)

class Tracker(object):
    def __init__(self):
        self.ttracks = np.array([])
        self.xcands = np.array([])
        self.xtracks = np.array([])

    def _update(self, xt, xd):
        obs = np.asarray(xd).reshape(-1, 1)
        #guess = self.xtracks.reshape(-1, 1)
        #self.xtracks, _ = scipy.cluster.vq.kmeans(
        #    obs, guess, iter=2, check_finite=False)
        #self.xtracks = sklearn.cluster.DBSCAN().fit(
        #self.xtracks = np.squeeze(np.transpose(self.xtracks))
        #self.xtracks = np.atleast_1d(self.xtracks)
        it_updated = np.full(xt.shape, False)
        im_used = np.full(xd.shape, False)
        thresh = 5.0**2
        XT = np.tile(xt, (xd.size, 1))
        XD = np.tile(obs, (1, xt.size))
        D = XT - XD
        outliers = D**2 > thresh
        inliers = np.logical_not(outliers)
        D[outliers] = 0.
        xt = xt + np.sum(D, axis=0)
        it_updated = np.any(inliers, axis=0)
        im_unused = np.logical_not(np.any(inliers, axis=1))
        return xt, it_updated, im_unused

    def update(self, t, xdetections):
        id_unused = np.full(xdetections.shape, True)
        if self.xtracks.size > 0:
            self.xtracks, it_updated, id_unused = \
                    self._update(self.xtracks, xdetections)
            xdetections = xdetections[id_unused]
            self.xtracks = self.xtracks[it_updated]
        if self.xcands.size > 0:
            self.xcands, ic_updated, id_unused = \
                    self._update(self.xcands, xdetections)
            self.xcands = self.xcands[ic_updated]
            xdetections = xdetections[id_unused]
            self.xtracks = np.hstack((self.xtracks,
                                      self.xcands))
        self.xcands = xdetections
        self.ttracks = np.full(self.xtracks.shape, t)

class ColorMap(object):
    def __init__(self, cm_name):
        plt_map = plt.get_cmap(cm_name)
        colors = plt_map.colors
        colors = [c + [1.] for c in colors]
        positions = np.linspace(0, 1, len(colors))
        self.map = pg.ColorMap(positions, colors)
    def __call__(self, x):
        return self.map.map(x)

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
            log.info('FPS: %4.1f kHz', self.fps / 1000.)
            self.tprint = tnow

class RadarPlotter(object):
    def __init__(self, iface, output='', size=(800,600),
                 two_pulse_cancel=False):
        # Data stuff
        self.tracker = Tracker()
        self.tstart = time.time()
        self.lock = threading.RLock()
        self.buf0 = np.array([])
        self.buf1 = np.array([])
        self.y0 = [0]
        self.two_pulse_cancel = two_pulse_cancel
        self.sif = np.array([])
        self.y1 = [0]
        self.x0 = [0]
        self.x1 = [0]
        self.cdets = ColorMap('viridis')
        self.tdets = np.array([])
        self.xdets = np.array([])
        self.ydets = np.array([])
        self.cfar = [0]
        self.range0 = np.array([0., 0.])
        self.range1 = np.array([0., 0.])
        # PyQtGraph stuff
        self.app = pg.Qt.QtGui.QApplication([])
        self.view = pg.GraphicsView()
        l = pg.GraphicsLayout(border=(100, 100, 100))
        self.view.setCentralItem(l)
        self.view.show()
        title_details = ''
        if self.two_pulse_cancel:
            title_details = '(with two-pulse cancel)'
        title = ' '.join(['RADAR Plotter', title_details])
        self.view.setWindowTitle(title)
        self.view.resize(*size)
        self.plt = []
        self.curve = []
        self.plt.append(l.addPlot(title="Trigger"))
        self.plt.append(l.addPlot(title="CPI Range Profile"))
        l.nextRow()
        l2 = l.addLayout(colspan=2)
        self.plt.append(l2.addPlot(title="Detections vs Time"))
        self.plt[0].showGrid(x=True, y=True)
        self.plt[1].showGrid(x=True, y=True)
        self.plt[2].showGrid(x=True, y=True)
        self.curve.append(self.plt[0].plot([], [], pen=(255,0,0)))
        self.curve.append(self.plt[1].plot([], [], pen=(255,0,0)))
        self.curve.append(self.plt[1].plot([], [], pen=(0,255,0)))
        self.curve.append(self.plt[1].plot([], [], pen=None, pxMode=True,
                                           symbolSize=3,
                                           symbol='o',
                                           symbolBrush=(255,255,0)))
        self.curve.append(self.plt[2].plot([], [], pen=None, pxMode=True,
                                           symbolSize=3,
                                           symbol='o',
                                           symbolBrush=(255,255,0)))
        self.curve.append(self.plt[2].plot([], [], pen=None, pxMode=True,
                                           symbolSize=10,
                                           symbol='o',
                                           symbolBrush=(255,0,0)))
        self.plt[1].setXRange(0, 400, padding=None)
        self.plt[1].setYRange(-20, 40, padding=None)
        self.plt[2].setYRange(15, 400, padding=None)
        # Audio stuff
        self.fps = FpsEstimator()
        if output:
            tee = interface.WaveFileTee(output, self.stream_callback)
            self.iface = iface(tee, init_callback=tee.set_framerate)
        else:
            self.iface = iface(self.stream_callback)
        self.chunksize = self.iface.chunksize
        self.FS = self.iface.framerate
        self.N = int(rti.Tp * self.FS)
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
                    sif = self.buf1[istart:iend]
                    # two-pulse cancelled
                    if self.sif.size > 0 and self.two_pulse_cancel:
                        icut = np.min([sif.size, self.sif.size])
                        sif_diff = sif[:icut] - self.sif[:icut]
                        self.x1, self.y1 = range_profile(sif_diff, self.N)
                        self.sif = sif[:icut]
                    else:
                        self.x1, self.y1 = range_profile(sif, self.N)
                        self.sif = sif
                    # CFAR
                    self.cfar = rti.threshold_cfar(self.y1, num_train=20,
                                                   num_guard=5, rate_fa=1.0E-1)
                    idets = self.y1 > self.cfar
                    tcurr = time.time()
                    xdets = self.x1[idets]
                    ydets = self.y1[idets]
                    tdets = np.full(xdets.shape, tcurr)
                    self.tdets = np.hstack((self.tdets, tdets))
                    self.xdets = np.hstack((self.xdets, xdets))
                    self.ydets = np.hstack((self.ydets, ydets))
                    self.tracker.update(tcurr, xdets)
                    itimeout = self.tdets > tcurr - 10.0
                    self.tdets = self.tdets[itimeout]
                    self.xdets = self.xdets[itimeout]
                    #self.ydets = self.ydets[itimeout]
                    # Grab the trigger signal + a couple extra frames
                    # for better visual on the rising/falling edge
                    istart = np.max([0, istart - 10])
                    iend = np.min([iend + 10, len(self.buf0)])
                    self.y0 = self.buf0[istart:iend]
                    self.x0 = np.arange(self.y0.shape[0])/float(self.FS)
                    # Clear the data buffers
                    self.buf0 = []
                    self.buf1 = []

    def updateplot(self):
        with self.lock:
            sig = rti.dbv(self.y1)
            colors = self.cdets(self.ydets).tolist()
            #pens = [pg.mkPen(color=c) for c in colors]
            pen=(255,255,0,0)
            cfar = rti.dbv(self.cfar)
            self.curve[0].setData(self.x0, self.y0)
            self.curve[1].setData(self.x1, sig)
            self.curve[2].setData(self.x1, cfar)
            #self.curve[3].setData(self.xdets, rti.dbv(self.ydets))
            self.curve[4].setData(self.tdets - self.tstart, self.xdets,
                                  symbolPen=pen)
            if self.tracker.xtracks.size > 0:
                print('>>>> upd tracker -----')
                print(self.tracker.xtracks.shape)
                print(self.tracker.ttracks.shape)
                self.curve[5].setData(self.tracker.ttracks - self.tstart,
                                      self.tracker.xtracks)
                print('<<<< upd tracker -----')
            self.range0[0] = np.min([self.range0[0], np.min(self.y0)])
            self.range0[1] = np.max([self.range0[0], np.max(self.y0)])
            self.plt[0].setYRange(*self.range0, padding=None)
            tcurr = time.time()
            tf = tcurr - self.tstart
            t0 = tf - 10.
            self.plt[2].setXRange(t0, tf, padding=None)

    def run(self):
        self.app.exec_()

def parse_args():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers()
    sps = {name: sp.add_parser(name) for name in
           ['playback', 'pyaudio']}
    sps['playback'].add_argument('input', help='input playback .wav file')
    sps['playback'].set_defaults(iface=make_playback)
    IDX = 3
    sps['pyaudio'].add_argument('-d', '--dev',
                                help='PyAudio device index (default: %d)' % IDX,
                                default=IDX)
    sps['pyaudio'].set_defaults(iface=make_device)
    for s in sps.values():
        s.add_argument('-o', '--output', help='output .wav file', default='')
        s.add_argument('-x', '--two-pulse-cancel',
                       help='Use two-pulse cancellation', action="store_true")
    return p.parse_args()

def make_playback(args):
    return lambda cb, **kwargs: interface.WaveFileReader(args.input, cb, **kwargs)

def make_device(args):
    return lambda cb, **kwargs: interface.PyAudioReader(args.dev, cb, **kwargs)

def main():
    fmt='%(asctime)s %(message)s'
    datefmt='%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.DEBUG,
                        format=fmt, datefmt=datefmt)
    args = parse_args()
    m = RadarPlotter(args.iface(args), output=args.output,
                     two_pulse_cancel=args.two_pulse_cancel)
    return m.run()

