"""
    Various tools for audio handing

    :authors: [Thomas Bagrel, Pierre Bouillon, Florian Vogt]
    :url: https://github.com/pBouillon/TELECOM_TAN
    :license: [MIT](https://github.com/pBouillon/TELECOM_TAN/blob/master/LICENSE)
"""

from collections import NamedTuple
from enum import Enum
from path import os

import numpy as np
import scipy.io as sio


"""Expected wav file rate
"""
CD_QUALITY_RATE = 44_100


class Phonem(Enum):
    """Phonem enum to classify phonems
    """
    A = 'a'
    E = 'e'
    E_ACUTE = 'é'
    E_AGRAVE = 'è'
    I = 'i'
    O = 'o'
    OU = 'ou'
    AN = 'an'
    IN = 'in'
    ON = 'on'
    U = 'u'


class RawSample(NamedTuple):
    """.wav data object
    """
    phonem: Phonem
    file_name: str
    data: np.array


def load_wav_file(file_path: str) -> RawSample:
    """Loads a wav file and returns the corresponding RawSample data object.

    :param file_path: path to the wav file

    :raise ValueError: On bad file rate (!= CD_QUALITY_RATE)
    :returns: the associated RawSample data object
    """
    # Read the .wav file
    rate, data = sio.wavfile.read(file_path)

    if rate != CD_QUALITY_RATE:
        raise ValueError(
            f'Invalid file rate, found {rate} Hz but '
            f'expected {CD_QUALITY_RATE} Hz')

    # Extract file meta data
    *_, phonem_str_caps, file_name = path.split(file_path)
    phonem = Phonem(phonem_str_caps.lower())

    # Instanciate the associated data object
    return RawSample(phonem, file_path, data)
