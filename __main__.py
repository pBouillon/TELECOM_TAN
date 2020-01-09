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
from utils.data_objects import Sample, Phoneme, Spectrum
from utils.sound_utils import wav_to_normalized_h_1, wav_to_normalized_h_2, h_2, \
    scalar_product, h_1
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
        audio.terminate()


BLANK_STD_THRESHOLD = 600


def is_blank(sample):
    std = np.std(sample)
    print(std)
    return std < BLANK_STD_THRESHOLD


def drop_blanks(record_gen):
    for i, sample in enumerate(record_gen):
        if i > 1 and not is_blank(sample):
            yield sample


def process_samples(record_gen):
    for sample in record_gen:
        yield h_2(Sample(
            phoneme=Phoneme.UNKNOWN,
            file_name='<audio input stream>',
            data=sample
        ))

from time import time
import os
from os import path

NEW_SAMPLES_STORE = "./assets/new"


def record_and_store():
    hs = []
    for i, h in enumerate(process_samples(drop_blanks(record()))):
        hs.append(h.data)
    hs = np.array(hs)
    h_avg = hs.mean(0)
    phoneme = input("Phoneme : ")
    if phoneme not in PHONEMES:
        print("\"{}\" n'est pas reconnu. Réessayez.".format(phoneme))
        phoneme = input("Phoneme : ")
    filename = "{}_{}".format(phoneme, int(time()))
    filepath = path.join(NEW_SAMPLES_STORE, filename)
    plt.plot(h_avg)
    plt.show()
    if input("Est-ce un bon échantillon ? (Y/N)") in 'YOyo':
        np.save(filepath, h_avg)
        print("Phonème enregistré sous le nom {}".format(filepath))
    else:
        print("Phonème non enregistré")


def load_samples():
    data_bank = []
    for r, d, f in os.walk(NEW_SAMPLES_STORE):
        for filepath in f:
            filename = path.basename(filepath)
            if filename.startswith("."):
                continue
            phoneme = filename.split("_")[0]
            assert(phoneme in PHONEMES)
            data_bank.append(Spectrum(
                data=np.load(path.join(NEW_SAMPLES_STORE,filepath)),
                freq=None,
                file_name=filepath,
                phoneme=Phoneme(phoneme)
            ))
    return data_bank


def main_2():
    data_bank = load_samples()

    while True:
        results = {}
        count = {}
        hs = []
        for phoneme in PHONEMES:
            results[phoneme] = 0.0
            count[phoneme] = 0

        for i, h in enumerate(process_samples(drop_blanks(record()))):
            hs.append(h.data)

        hs = np.array(hs)
        h_avg = hs.mean(0)
        
        best_result = 0
        best_hh_data = None
        best_phoneme = None
        for hh in data_bank:
            score = float(scalar_product(h_avg, hh.data))
            print(h_avg.shape, hh.data.shape, type(score), score)
            print(hh.file_name)
            if score > best_result:
                best_hh_data = hh.data
                best_result = score
                best_phoneme = hh.phoneme.value

        print("Identified: {} with score {}".format(best_phoneme, best_result))
        plt.plot(np.arange(len(h_avg)), h_avg, '-r', np.arange(len(h_avg)), best_hh_data, '-.b')
        plt.show()




def main_1():
    # fig, axs = plt.subplots(3, 4)

    data_bank = []

    for i, phoneme in enumerate(PHONEMES):
        # ax = axs[i // 4, i % 4]
        # ax.set_title(f'Phonème {phoneme}')

        for pitch in PITCHES:
            for author in AUTHORS:
                played_file = f'./assets/{phoneme}_{author["name"]}_{pitch["num"]}.wav'
                print(f'# Phonème {phoneme} par {author["name"]} : pitch {pitch["num"]}')
                h = wav_to_normalized_h_2(played_file)
                # ax.plot(h.freq, h.data, linestyle=author["style"], color=pitch["color"])
                data_bank.append(h)

    # plt.show()

    while True:
        results = {}
        count = {}
        hs = []
        for phoneme in PHONEMES:
            results[phoneme] = 0.0
            count[phoneme] = 0

        for i, h in enumerate(process_samples(drop_blanks(record()))):
            hs.append(h.data)
            for hh in data_bank:
                results[hh.phoneme.value] += scalar_product(h, hh) ** 2
                count[hh.phoneme.value] += 1

        for phoneme in PHONEMES:
            if count[phoneme] != 0:
                results[phoneme] /= count[phoneme]

        best_result = 0
        best_phoneme = None
        for phoneme in PHONEMES:
            if results[phoneme] > best_result:
                best_phoneme = phoneme
                best_result = results[phoneme]

        print("Identified: {} with score {}".format(best_phoneme, best_result))
        print(results)

        hs = np.array(hs)
        h_avg = hs.mean(0)
        plt.plot(h_avg)
        plt.show()


def main_main():
    method = input("""Please enter your choice : 
    (1) Use assets base
    (2) Use live (recorded) base 
    (3) Add sound to the live base\n""")

    if method == '1':
        main_1()
    else:
        if method == '2':
            main_2()
        else:
            if method == '3':
                while(True):
                    record_and_store()


if __name__ == "__main__":
    main_main()
