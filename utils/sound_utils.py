"""
    Various tools for audio handing

    :authors: [Thomas Bagrel, Pierre Bouillon, Florian Vogt]
    :url: https://github.com/pBouillon/TELECOM_TAN
    :license: [MIT](https://github.com/pBouillon/TELECOM_TAN/blob/master/LICENSE)
"""

from math import pi

import scipy.io.wavfile as wavfile
import numpy as np
from pathlib2 import Path

from utils.constants import N, CD_QUALITY_RATE, NSV, K_MIN, K_MAX, \
    T_E, KV, K_C, RO, QSV, Q
from utils.data_objects import Sample, Phoneme, Spectrum


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
    file_name = Path(file_path).name
    raw_phoneme = file_name.split('_')[0]
    try:
        phoneme = Phoneme(raw_phoneme.lower())
    except ValueError:
        raise ValueError(f'Invalid phoneme "{raw_phoneme.lower()}"')

    # Instantiate the associated data object
    return Sample(phoneme, file_name, data)


def window(sample: Sample) -> Sample:
    """Fetch the hamming window for the current sample

    :param sample: audio sample to work on

    :returns: the sample with the the Hamming window applied on its data
    """
    hamming_window = (1 / 2) * (1 + np.cos(2 * pi * (NSV - N / 2) / N))
    return Sample(
        phoneme=sample.phoneme,
        file_name=sample.file_name,
        data=hamming_window * sample.data
    )


def spectrum_of(sample: Sample) -> Spectrum:
    """Apply the discrete Fourier Transform on the current sample

    :remark: Spectrum.data only contains the modulus of the fft spectrum

    :param sample: audio sample to work on

    :returns: the generated Spectrum object
    """
    data = np.concatenate(
        [np.zeros(K_MIN),
         np.abs(np.fft.fft(sample.data))[K_MIN:K_MAX]]
    )

    freq = np.fft.fftfreq(N, T_E)[:K_MAX]

    return Spectrum(
        data=data,
        file_name=sample.file_name,
        freq=freq,
        phoneme=sample.phoneme
    )


def enhance_high_freqs(spectrum: Spectrum) -> Spectrum:
    """TODO
    """
    p = np.sqrt(1 + (KV / K_C) ** 2)
    data = spectrum.data * p

    return Spectrum(
        data=data,
        file_name=spectrum.file_name,
        freq=spectrum.freq,
        phoneme=spectrum.phoneme
    )


def biased_log(spectrum: Spectrum) -> (Spectrum, float):
    """TODO
    """
    avg = np.average(spectrum.data)
    beta = RO * avg
    data = np.log(spectrum.data + beta)

    return Spectrum(
        data=data,
        file_name=spectrum.file_name,
        freq=spectrum.freq,
        phoneme=spectrum.phoneme
    ), beta


def smooth(spectrum: Spectrum) -> Spectrum:
    """TODO
    """
    weight_func = (1 / 2) * (1 + np.cos(2 * pi * (QSV - Q / 2) / Q))
    total = np.sum(weight_func)
    weight_func = weight_func / total
    data = np.convolve(spectrum.data, weight_func, mode='same')

    return Spectrum(
        data=data,
        file_name=spectrum.file_name,
        freq=spectrum.freq,
        phoneme=spectrum.phoneme
    )


def biased_exp(spectrum: Spectrum, beta: float) -> Spectrum:
    """TODO
    """
    data = np.exp(spectrum.data) - beta

    return Spectrum(
        data=data,
        file_name=spectrum.file_name,
        freq=spectrum.freq,
        phoneme=spectrum.phoneme
    )
