"""
    Program's entry-point

    :authors: [Thomas Bagrel, Pierre Bouillon, Florian Vogt]
    :url: https://github.com/pBouillon/TELECOM_TAN
    :license: [MIT](https://github.com/pBouillon/TELECOM_TAN/blob/master/LICENSE)
"""

from sys import argv

import matplotlib.pyplot as plt

from utils.sound_utils import load_wav_file, window, spectrum_of, \
    enhance_high_freqs, biased_log, smooth, biased_exp


"""Default audio file to load
"""
DEFAULT_PLAYED_FILE = './assets/a_fvogt_1.wav'


def main():
    # select played file
    played_file = DEFAULT_PLAYED_FILE if len(argv) == 1 else argv[1]

    # extract .wav data
    raw_sample = load_wav_file(played_file)
    w = window(raw_sample)
    s = spectrum_of(w)
    h = enhance_high_freqs(s)
    l, beta = biased_log(h)
    ss = smooth(l)
    e = biased_exp(ss, beta)

    # result display
    plt.title(f'Spectrum of "{ss.file_name}"')
    plt.plot(ss.freq, ss.data, ss.freq, l.data)
    plt.show()


if __name__ == "__main__":
    main()
