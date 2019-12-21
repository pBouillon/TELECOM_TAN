"""
    Data objects to handle extracted data

    :authors: [Thomas Bagrel, Pierre Bouillon, Florian Vogt]
    :url: https://github.com/pBouillon/TELECOM_TAN
    :license: [MIT](https://github.com/pBouillon/TELECOM_TAN/blob/master/LICENSE)
"""

from enum import Enum
from typing import NamedTuple

import numpy as np


class Phoneme(Enum):
    """Phoneme enum to classify phonemes
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

    """Handled phoneme
    """
    phoneme: Phoneme

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

    """Handled Phoneme
    """
    phoneme: Phoneme
