"""
    Various tools for audio handing

    :authors: [Thomas Bagrel, Pierre Bouillon, Florian Vogt]
    :url: https://github.com/pBouillon/TELECOM_TAN
    :license: [MIT](https://github.com/pBouillon/TELECOM_TAN/blob/master/LICENSE)
"""

from typing import NamedTuple
from enum import Enum
from os import path

import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt


"""Expected wav file rate
"""
CD_QUALITY_RATE = 44_100

"""Power of 2 for N
"""
NU = 13

"""Number of data points in each samples
"""
N = 2 ** NU

"""Sampling period
"""
T_E = 1 / CD_QUALITY_RATE

"""Sample's duration
"""
T = N * T_E

"""Time vector
"""
TV = np.linspace(0, T, N, endpoint=False)


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


class RawSample(NamedTuple):
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


def load_wav_file(file_path: str) -> RawSample:
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
    return RawSample(phonem, file_path, data)

