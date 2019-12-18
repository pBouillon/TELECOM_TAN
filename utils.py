"""
    Various tools for audio handing

    :authors: [Thomas Bagrel, Pierre Bouillon, Florian Vogt]
    :url: https://github.com/pBouillon/TELECOM_TAN
    :license: [MIT](https://github.com/pBouillon/TELECOM_TAN/blob/master/LICENSE)
"""

from enum import Enum
from typing import NamedTuple
from os import path
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile


"""Expected wav file rate
"""
CD_QUALITY_RATE = 44_100

"""Power of 2 for N
"""
NU = 13

"""Number of data points in each samples
"""
N = 2 ** NU

"""Sample spacing
"""
T_E = 1 / CD_QUALITY_RATE

"""Sample's duration
"""
T = N * T_E

"""Time vector
"""
TV = np.linspace(-T / 2, T / 2, N, endpoint=False)

"""N-shifted vector 
"""
NSV = np.arange(0, N, 1)

class Phonem(Enum):
    """Phonem enum to classify phonems
    """
    A = 'a'
    AN = 'an'
    E = 'e'
    E_ACUTE = 'é'
    E_AGRAVE = 'è'
    I = 'i'
    IN = 'in'
    O = 'o'
    ON = 'on'
    OU = 'ou'
    U = 'u'


class Sample(NamedTuple):
    """.wav data object
    """

    """Handled phonem
    """
    phonem: Phonem

    """Source of the recording
    """
    file_name: str

    """Data read from wav file
    """
    data: np.array


class Spectrum(NamedTuple):
    """TODO
    """
    
    """Sample value
    """
    data: np.array
    
    """Sample frequency
    """
    freq: np.array
    
    """Source of the original recording
    """
    file_name: str
    
    """Handled Phonem
    """
    phonem: Phonem


def spectrum(sample: Sample) -> Spectrum:
    """Apply the discrete Fourier Transform on the current sample

    :remark: Spectrum.data only contains the modulus of the fft spectrum

    :param sample: audio sample to work on

    :returns: the generated Spectrum object
    """
    data = np.abs(np.fft.fft(sample.data))
    freq = np.fft.fftfreq(N, T_E)
    return Spectrum(
        data=data,
        file_name=sample.file_name,
        freq=freq,
        phonem=sample.phonem
    )


def load_wav_file(file_path: str) -> Sample:
    """Loads a wav file and returns the corresponding RawSample data object.

    :param file_path: path to the wav file

    :raise ValueError: on bad file rate (!= CD_QUALITY_RATE)
    :returns: the associated RawSample data object
    """
    # Read the .wav file
    rate, data = wavfile.read(file_path)

    # cut the number of data points to the chosen power of 2
    data = np.array(data[:N])

    if rate != CD_QUALITY_RATE:
        raise ValueError(
            f'Invalid file rate, found {rate} Hz but '
            f'expected {CD_QUALITY_RATE} Hz')

    # Extract file meta data
    phonem_str_caps, file_name = file_path.replace('\\', '/').split('/')[-2:]
    phonem = Phonem(phonem_str_caps.lower())

    # Instanciate the associated data object
    return Sample(phonem, file_path, data)


def window(sample: Sample) -> Sample:
    """Fetch the hamming window for the current sample

    :param sample: audio sample to work on

    :returns: the sample with the the Hamming window applied on its data
    """
    hamming_window = (1 / 2) * (1 + np.cos(2 * pi * (NSV - N / 2) / N))
    return Sample(
        phonem=sample.phonem,
        file_name=sample.file_name,
        data=hamming_window * sample.data
    )


def main():
    raw_sample = load_wav_file("/home/thomas/Bureau/samples/A/a_fvogt_1.wav")
    w = window(raw_sample)
    s = spectrum(w)
    plt.plot(s.freq, s.data)
    plt.show()


if __name__ == "__main__":
    main()
