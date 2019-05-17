# coffee-can-radar

Interface software for the MIT IAP "coffee can" RADAR.

Adapted from the course's MATLAB post-processing scripts.

`candar` is a real-time plotting utility for displaying
both left/right channels (CPI trigger channel and the
radar CPI channel), leveraging `pyqtgraph`.

# Hardware Dependencies

* A RADAR system as documented on the [MIT IAP course](https://ocw.mit.edu/resources/res-ll-003-build-a-small-radar-system-capable-of-sensing-range-doppler-and-synthetic-aperture-radar-imaging-january-iap-2011/)
  * In general, can be any radar system that interfaces to the processing
    via stereo audio, with L=trigger and R=pulse signal. Some tuning of
    software may be required for triggers not at 20 ms intervals (some
    stuff is still hardcoded).
  * Many laptops don't record stereo audio through the audo port itself
    (microphones are normally mono, e.g., Apple Macbook Pros use the
    3.5mm audio for stereo output + mono input). You can purchase USB
    audio adapters to convert the IAP course's single audio connector
    output into a USB input suitable for a laptop ([example](https://www.amazon.com/gp/product/B000KW2YEI/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&psc=1)).

# Software Dependencies

* Python 3
* Numpy
* Scipy
* Matplotlib
* PyAudio
* PyQtGraph

# Quick Start

To read from PyAudio device:
```
$ ./candar pyaudio
```

To read from recorded .wav file:
```
$ ./candar playback input.wav
```

For full list of options:
```
$ ./candar -h
```

