import logging
import threading
import time
import wave
import pyaudio

log = logging.getLogger(__name__)

class WaveFileTee(object):
    def __init__(self, filename, callback):
        self.wf = wave.open(filename, 'wb')
        self.wf.setnchannels(2)
        self.wf.setsampwidth(2)
        self.callback = callback

    def set_framerate(self, framerate):
        self.wf.setframerate(framerate)

    def __call__(self, *args, **kwargs):
        self.callback(*args, **kwargs)
        self.wf.writeframes(args[0])

    def __del__(self):
        self.wf.close()

class WaveFileReader(object):
    def __init__(self, filename, callback, chunksize=1024,
                 init_callback=None):
        self.wf = wave.open(filename, 'r')
        self.framerate = self.wf.getframerate()
        self.chunksize = chunksize
        self.period = float(self.chunksize) / self.wf.getframerate()
        self.callback = callback
        self.bgt = []
        log.info('Opened \'%s\': %s', filename, str(self.wf.getparams()))
        if init_callback:
            init_callback(self.framerate)

    def _playback_handler(self):
        while True:
            t_start = time.time()
            in_data = self.wf.readframes(self.chunksize)
            if not in_data:
                break
            t_tgt = t_start + self.period
            self.callback(in_data, self.chunksize, {}, None)
            t_now = time.time()
            t_sleep = max([0., t_tgt - t_now])
            time.sleep(t_sleep)

    def start(self):
        self.bgt.append(threading.Thread(target=self._playback_handler))
        self.bgt[0].start()

class PyAudioReader(object):
    def __init__(self, device_index, callback, chunksize=1024,
                 init_callback=None):
        self.framerate = 48000  # TODO read from device params
        self.callback = callback
        self.chunksize = chunksize
        self.device_index = device_index
        self.nchannels = 2
        self.bgt = []
        self.p = pyaudio.PyAudio()
        if init_callback:
            init_callback(self.framerate)

    def _stream_handler(self, *args, **kwargs):
        self.callback(*args, **kwargs)
        return None, pyaudio.paContinue

    def start(self):
        self.bgt.append(self.p.open(input=True,
                               input_device_index=self.device_index,
                               format=pyaudio.paInt16,
                               channels=self.nchannels,
                               rate=self.framerate,
                               frames_per_buffer=self.chunksize,
                               stream_callback=self._stream_handler))

