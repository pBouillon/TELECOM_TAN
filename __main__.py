"""
    Program's entry-point

    :authors: [Thomas Bagrel, Pierre Bouillon, Florian Vogt]
    :url: https://github.com/pBouillon/TELECOM_TAN
    :license: [MIT](https://github.com/pBouillon/TELECOM_TAN/blob/master/LICENSE)
"""

import matplotlib.pyplot as plt

from utils.sound_utils import wav_to_normalized_h_1, wav_to_normalized_h_2

"""Default audio file to load
"""
DEFAULT_PLAYED_FILE = './assets/o_pbouillon_3.wav'

PHONEMES = [
    "a", "an", "e", "i", "in", "o", "on", "ou", "u", "è", "é"
]

AUTHORS = [
    {"name": "pbouillon", "style": "-."},
    {"name": "fvogt", "style": "-"},
    # {"name": "tbagrel", "style": "--"}
]

PITCHES = [
    {"num": 1, "color": "orangered"},
    {"num": 2, "color": "red"},
    {"num": 3, "color": "brown"}
]


def main():
    fig, axs = plt.subplots(3, 4)

    for i, phoneme in enumerate(PHONEMES):
        ax = axs[i // 4, i % 4]
        ax.set_title(f'Phonème {phoneme}')

        for pitch in PITCHES:
            for author in AUTHORS:
                played_file = f'./assets/{phoneme}_{author["name"]}_{pitch["num"]}.wav'
                n = wav_to_normalized_h_1(played_file)
                ax.plot(n.freq, n.data, linestyle=author["style"], color=pitch["color"])

    plt.show()


def main_2():
    wav_to_normalized_h_2(DEFAULT_PLAYED_FILE)


if __name__ == "__main__":
    main_2()
