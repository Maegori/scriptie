
import os
import essentia
essentia.log.warningActive = False
essentia.log.infoActive = False

import essentia.standard as es
import vamp
import numpy as np
from scipy.signal import medfilt
from synthesizer import Player, Synthesizer, Waveform

#Setup audio stream

class Notify():
    def __init__(self):
        self.rythm_extractor = es.RhythmExtractor2013()

        self.wind = es.Windowing(type='hann')
        self.spec = es.Spectrum()
        self.mfcc = es.MFCC()
        self.sbic = es.SBic(minLength=3)   

        # self.pitches = []
        # self.time = 0
        # self.note_time = 0
        # self.note_time_dt = 0
        # self.audio_reset = False
        # self.audio_pause = True

        # self.sample_rate = 44100
        # self.note_dt = 2000        #Num Samples
        # self.note_duration = 20000 #Num Samples
        # self.note_decay = 5.0 / self.sample_rate

        self.pitch_values = []
        # self.PYA = pyaudio.PyAudio()

        # self.audio_stpream = PYA.open(
        #     format = PYA.get_format_from_width(2),
        #     channels=1,
        #     rate=self.sample_rate,
        #     output=True,
        #     stream_callback=self.audio_callback)

        # self.audio_stream.start_stream()

    def segment(self, audio):
        pool = essentia.Pool()

        for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
            _, mfcc_coeffs = self.mfcc(self.spec(self.wind(frame)))
            pool.add('lowlevel.mfcc', mfcc_coeffs)

        segments = self.sbic(pool['lowlevel.mfcc'].T)

        return 0 if len(segments) == 2 else int(segments[1])

    def audio_to_notes(self, input_fn):
        fs = 44100
        hop = 128
        audio = es.MonoLoader(filename=input_fn, downmix='mix', sampleRate=44100)()
        start = self.segment(audio)
        audio = audio[start:]
          
        params = {"minfqr": 100.0, "maxfqr": 800.0, "voicing": 0.2, "minpeaksalience": 0.0}
        data = vamp.collect(audio, fs, "mtg-melodia:melodia", parameters=params)
        hop, pitch_values = data['vector']
        timestamps = 8 * 128/44100.0 + np.arange(len(pitch_values)) * (128 / 44100.0)
        pitch_values = np.absolute(np.array(pitch_values))

        print(len(timestamps))

        wav = []

        # wave_output = wave.open('test.wav', 'wb')
        # wave_output.setparams((1,2, 44100, 0, 'NONE', 'not compressed'))

        dt = 128/44100


        player = Player()
        player.open_stream()
        synthesizer = Synthesizer(osc1_waveform=Waveform.sine, osc1_volume=1.0, use_osc2=False)
        print("playing sound")
        for f in pitch_values:
            player.play_wave(synthesizer.generate_constant_wave(f, dt))




        # bpm = self.rythm_extractor(audio)[0]

notes = Notify()

filename = "data/2020/Iceland_Think About Things_Dai & Gagnamagni.mp3"
output = "test.mid"

notes.audio_to_notes(filename)