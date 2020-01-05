"""
    Various tools for audio handing

    :authors: [Thomas Bagrel, Pierre Bouillon, Florian Vogt]
    :url: https://github.com/pBouillon/TELECOM_TAN
    :license: [MIT](https://github.com/pBouillon/TELECOM_TAN/blob/master/LICENSE)
"""

from math import pi

import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
from pathlib2 import Path

from utils.constants import N, CD_QUALITY_RATE, NSV, K_MIN, K_MAX, \
    T_E, KV, K_C, RO, QSV, Q
from utils.data_objects import Sample, Phoneme, Spectrum, Cepstrum


def load_wav_file(file_path: str) -> Sample:
    """Loads a wav file and returns the corresponding Sample data object.

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


def windowed(sample: Sample) -> Sample:
    """Applies the hamming windowed on the current sample.

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
    """Applies the discrete Fourier Transform on the current sample.
    Frequencies above K_MAX / T (about 3000 Hz) are cut.

    :remark: Spectrum.data only contains the modulus of the fft spectrum

    :param sample: audio sample to work on

    :returns: the generated Spectrum object
    """
    data = np.abs(np.fft.fft(sample.data))[:K_MAX]

    freq = np.fft.fftfreq(N, T_E)[:K_MAX]

    return Spectrum(
        data=data,
        file_name=sample.file_name,
        freq=freq,
        phoneme=sample.phoneme
    )


def zero_low_frequencies(spectrum: Spectrum) -> Spectrum:
    """TODO
    """

    data = np.concatenate([np.zeros(K_MIN), spectrum.data[K_MIN:]])

    return Spectrum(
        data=data,
        file_name=spectrum.file_name,
        freq=spectrum.freq,
        phoneme=spectrum.phoneme
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
    # normalize weight function
    weight_func = (1 / 2) * (1 + np.cos(2 * pi * (QSV - Q / 2) / Q))
    total = np.sum(weight_func)
    weight_func = weight_func / total

    # smooth the signal
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


def normalize(spectrum: Spectrum) -> Spectrum:
    """TODO
    """

    data = np.copy(spectrum.data)
    data -= np.mean(data)
    data /= np.std(data)

    return Spectrum(
        data=data,
        file_name=spectrum.file_name,
        freq=spectrum.freq,
        phoneme=spectrum.phoneme
    )


def scalar_product(spectrum_a: Spectrum, spectrum_b: Spectrum) -> float:
    """TODO
    """

    return np.dot(spectrum_a.data, spectrum_b.data)


def wav_to_normalized_h_1(played_file: str) -> Spectrum:
    """TODO
    """

    raw_sample = load_wav_file(played_file)
    w = windowed(raw_sample)
    s = spectrum_of(w)
    s = enhance_high_freqs(s)
    s = zero_low_frequencies(s)

    l, beta = biased_log(s)
    sl = smooth(l)
    s = biased_exp(sl, beta)
    n = normalize(s)

    return n


def wav_to_normalized_h_2(played_file: str) -> Spectrum:
    """TODO
    """

    raw_sample = load_wav_file(played_file)
    w = windowed(raw_sample)
    s = np.fft.fft(w.data)
    p = np.sqrt(1 + (KV / K_C) ** 2)
    pp = np.concatenate([p, np.ones(len(s) - 2 * K_MAX), p[::-1]])
    s = s * pp

    f = np.fft.fftfreq(len(w.data), T_E)[:K_MAX]
    n = len(s)

    c, ndelay = complex_cepstrum(s)
    rc = real_cepstrum(s)
    K_C_MIN_SEARCH = int(np.round(1 / (200 * T_E)))
    K_C_MAX_SEARCH = int(np.round(1 / (75 * T_E)))
    k_f_p = np.argmax(rc[K_C_MIN_SEARCH:K_C_MAX_SEARCH]) + K_C_MIN_SEARCH

    f_p = 1 / (k_f_p * T_E)
    print(f_p)

    k_f_p -= 50
    print(k_f_p)

    c_altered = np.concatenate([c[:k_f_p], np.zeros(len(c) - 2 * k_f_p), c[-k_f_p:]])

    print(len(c))
    assert(len(c) == len(c_altered))

    new_s = inverse_complex_cepstrum(c_altered, ndelay)

    return normalize(Spectrum(
        data=np.concatenate([np.zeros(K_MIN), np.abs(new_s[K_MIN:K_MAX])]),
        freq=f,
        file_name=raw_sample.file_name,
        phoneme=raw_sample.phoneme
    ))


def complex_cepstrum(spectrum):
    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = np.unwrap(phase)
        center = (samples + 1) // 2
        if samples == 1:
            center = 0
        ndelay = np.array(np.round(unwrapped[..., center] / np.pi))
        unwrapped -= np.pi * ndelay[..., None] * np.arange(samples) / center
        return unwrapped, ndelay

    unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
    log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped_phase
    ceps = np.fft.ifft(log_spectrum).real

    return ceps, ndelay



def real_cepstrum(spectrum):
    ceps = np.fft.ifft(np.log(np.abs(spectrum))).real

    return ceps



def inverse_complex_cepstrum(ceps, ndelay):

    def _wrap(phase, ndelay):
        ndelay = np.array(ndelay)
        samples = phase.shape[-1]
        center = (samples + 1) // 2
        wrapped = phase + np.pi * ndelay[..., None] * np.arange(samples) / center
        return wrapped

    log_spectrum = np.fft.fft(ceps)
    spectrum = np.exp(log_spectrum.real + 1j * _wrap(log_spectrum.imag, ndelay))
    return spectrum
