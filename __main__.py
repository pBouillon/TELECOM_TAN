"""
    Program's entry-point

    :authors: [Thomas Bagrel, Pierre Bouillon, Florian Vogt]
    :url: https://github.com/pBouillon/TELECOM_TAN
    :license: [MIT](https://github.com/pBouillon/TELECOM_TAN/blob/master/LICENSE)
"""

import matplotlib.pyplot as plt
import pyaudio
import numpy as np

from utils.constants import N
from utils.data_objects import Sample, Phoneme
from utils.sound_utils import wav_to_normalized_h_1, wav_to_normalized_h_2, h_2, \
    scalar_product
from utils.easy_thread import ThreadPool, DebugLevel

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

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


def record():
    input("Press enter to start recording...")
    stop = False

    def ask_for_stop():
        nonlocal stop
        input("Press enter to stop recording...")
        stop = True

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT, channels=CHANNELS,
        rate=RATE, input=True,
        frames_per_buffer=N)

    ThreadPool(debug_level=DebugLevel.ERROR).add(1, ask_for_stop)

    dt = np.dtype("i2")

    while not stop:
        data = np.frombuffer(stream.read(N), dtype=dt)
        yield data


BLANK_STD_THRESHOLD = 8000


def is_blank(sample):
    std = np.std(sample)
    print(std)
    return std < BLANK_STD_THRESHOLD


def drop_blanks(record_gen):
    for i, sample in enumerate(record_gen):
        if i > 0 and not is_blank(sample):
            yield sample


def process_samples(record_gen):
    for sample in record_gen:
        yield h_2(Sample(
            phoneme=Phoneme.UNKNOWN,
            file_name='<audio input stream>',
            data=sample
        ))


def main():
    fig, axs = plt.subplots(3, 4)

    data_bank = []

    for i, phoneme in enumerate(PHONEMES):
        ax = axs[i // 4, i % 4]
        ax.set_title(f'Phonème {phoneme}')

        for pitch in PITCHES:
            for author in AUTHORS:
                played_file = f'./assets/{phoneme}_{author["name"]}_{pitch["num"]}.wav'
                print(f'# Phonème {phoneme} par {author["name"]} : pitch {pitch["num"]}')
                h = wav_to_normalized_h_2(played_file)
                # ax.plot(h.freq, h.data, linestyle=author["style"], color=pitch["color"])
                data_bank.append(h)

    results = {}
    for phoneme in PHONEMES:
        results[phoneme] = 0.0

    for i, h in enumerate(process_samples(drop_blanks(record()))):
        for hh in data_bank:
            results[hh.phoneme.value] += scalar_product(h, hh) ** 2

    best_result = 0.0
    best_phoneme = None
    for phoneme in PHONEMES:
        if results[phoneme] > best_result:
            best_phoneme = phoneme
            best_result = results[phoneme]

    print("Identified: {} with score {}".format(best_phoneme, best_result))


if __name__ == "__main__":
    main()
