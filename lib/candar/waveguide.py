import numpy as np

def waveguide_lambda_inches(f0, d):
    # f0 = Operating frequency (MHz)
    # d = diameter of can (inches)
    # Cutoff frequency of TE11 (MHz)
    fc = 6917.26 / d
    # Standing wavelength in guide (inches)
    Wg = 11802.85 / np.sqrt(f0**2 - fc**2)

    c = 3.0e8  # m/s
    lc = c / (fc * 1.0e6)  # m
    d2 = lc / 1.705  # m
    print('d: %f inches, d2: %f inches' % (d, d2 * 100./2.54))
    return Wg

f0 = 2349.
f0 = 2270.
f1 = 2472.
d = 4.0

Wg0 = waveguide_lambda_inches(f0, d)
Wg1 = waveguide_lambda_inches(f1, d)

print('Quarter waveguide f0=%6.1f MHz: %4.2f' % (f0, Wg0 / 4.))
print('Quarter waveguide f0=%6.1f MHz: %4.2f' % (f1, Wg1 / 4.))


