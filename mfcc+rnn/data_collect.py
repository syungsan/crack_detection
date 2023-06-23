#!/usr/bin/env python
# coding: utf-8

import os
import sounddevice as sd
import soundfile as sf
import datetime


duration = 10  # 10秒間録音する
samplerate = 16000


def set_input_device():

    print(sd.query_devices())
    devices = sd.default.device

    selected = input("\nplease select recording device id.\n")

    if selected != "" and selected.isdecimal():
        devices[0] = int(selected)
        sd.default.device = devices

    print(sd.default.device)

def recording(file_path):

    print("now recording...")

    # 録音
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait() # 録音終了待ち

    print(recording.shape) #=> (duration * sr_in, channels)

    # 録音信号のNumPy配列をwav形式で保存
    sf.write(file_path, recording, samplerate)


def main():

    os.makedirs("../temp/record", exist_ok=True)

    set_input_device()

    now = datetime.datetime.now()
    recording("../temp/record/{}.wav".format(now.strftime("%Y%m%d_%H%M%S")))

    print("\nall process was completed...")


if __name__ == "__main__":
    main()
