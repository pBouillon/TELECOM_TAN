"""
    Various constants used for audio handling

    :authors: [Thomas Bagrel, Pierre Bouillon, Florian Vogt]
    :url: https://github.com/pBouillon/TELECOM_TAN
    :license: [MIT](https://github.com/pBouillon/TELECOM_TAN/blob/master/LICENSE)
"""

from math import floor

import numpy as np


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

"""Shifted N vector
"""
NSV = np.arange(0, N, 1)

"""TODO
"""
K_MAX = 512

"""TODO
"""
TARGET_F_MIN = 300

"""TODO
"""
K_MIN = floor(TARGET_F_MIN * T)

"""TODO
"""
F_MAX = K_MAX / T

"""TODO
"""
KV = np.arange(0, K_MAX, 1)

"""TODO
"""
K_C = 80  # TODO: adjust

"""TODO
"""
RO = .01

"""TODO
"""
F_P_REF = 150

"""TODO
"""
Q = floor(F_P_REF * T)

"""TODO
"""
QSV = np.arange(0, Q, 1)
