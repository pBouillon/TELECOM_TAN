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
N_2 = 2 ** NU
N = N_2 * 4

"""Sample spacing
"""
T_E = 1 / CD_QUALITY_RATE

"""Sample's duration
"""
T = N_2 * T_E

"""Time vector
"""
TV = np.linspace(-T / 2, T / 2, N_2, endpoint=False)

"""Shifted N vector
"""
NSV = np.arange(0, N_2, 1)

"""TODO
"""
K_MAX = 512*4

"""TODO
"""
TARGET_F_MIN = 300

"""TODO
"""
K_MIN = floor(TARGET_F_MIN * T)*4

"""TODO
"""
F_MAX = K_MAX / (4*T)

"""TODO
"""
KV = np.arange(0, K_MAX, 1)

"""TODO
"""
K_C = 80*4  # TODO: adjust

"""TODO
"""
RO = .01

"""TODO
"""
F_P_REF = 150

"""TODO
"""
Q = floor(F_P_REF * T)*4

"""TODO
"""
QSV = np.arange(0, Q, 1)
