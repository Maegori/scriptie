import os

import essentia

essentia.log.warningActive = False
essentia.log.infoActive = False

import essentia.standard as es
import vamp
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# from essentia.standard import *
from midiutil.MidiFile import MIDIFile
from scipy.signal import medfilt


class Preprocessor():
    def __init__(self):
        #initiate algorithms
        self.pitch_extractor = es.PredominantPitchMelodia(frameSize=1024, hopSize=128, guessUnvoiced=True)
        self.rythm_extractor = es.RhythmExtractor2013()

        self.wind = es.Windowing(type='hann')
        self.spec = es.Spectrum()
        self.mfcc = es.MFCC()
        self.sbic = es.SBic(minLength=3)

    def save_midi(self, output_fn, notes, tempo):

        track = 0
        time = 0
        midifile = MIDIFile(1)

        midifile.addTrackName(track, time, "MIDI TRACK")
        midifile.addTempo(track, time, tempo)

        channel = 0
        volume = 100

        for note in notes:
            onset = note[0] * (tempo/60.)
            duration = note[1] * (tempo/60.)
            # duration = 1
            pitch = int(note[2])
            # print(pitch)
            midifile.addNote(track, channel, pitch, onset, duration, volume)

        with open(output_fn, "wb") as f:
            midifile.writeFile(f)

    def midi_to_notes(self, midi, fs, hop, smooth, minduration):

        # smooth midi pitch sequence first
        if (smooth > 0):
            filter_duration = smooth  # in seconds
            filter_size = int(filter_duration * fs / float(hop))
            if filter_size % 2 == 0:
                filter_size += 1
            midi_filt = medfilt(midi, filter_size)
        else:
            midi_filt = midi
        # print(len(midi),len(midi_filt))

        notes = []
        p_prev = 0
        duration = 0
        onset = 0
        for n, p in enumerate(midi_filt):
            if p == p_prev:
                duration += 1
            else:
                # treat 0 as silence
                if p_prev > 0:
                    # add note
                    duration_sec = duration * hop / float(fs)
                    # only add notes that are long enough
                    if duration_sec >= minduration:
                        onset_sec = onset * hop / float(fs)
                        notes.append((onset_sec, duration_sec, p_prev))

                # start new note
                onset = n
                duration = 1
                p_prev = p

        # add last note
        if p_prev > 0:
            # add note
            duration_sec = duration * hop / float(fs)
            onset_sec = onset * hop / float(fs)
            notes.append((onset_sec, duration_sec, p_prev))

        return notes


    def hz2midi(self, hz):

        # convert from Hz to midi note
        hz_nonneg = hz.copy()
        idx = hz_nonneg <= 0
        hz_nonneg[idx] = 1
        midi = 69 + 12*np.log2(hz_nonneg/440.)
        midi[idx] = 0

        # round
        midi = np.round(midi)

        return midi

    def segment(self, audio):
        pool = essentia.Pool()

        for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
            _, mfcc_coeffs = self.mfcc(self.spec(self.wind(frame)))
            pool.add('lowlevel.mfcc', mfcc_coeffs)

        segments = self.sbic(pool['lowlevel.mfcc'].T)

        return 0 if len(segments) == 2 else int(segments[1])


    def audio_to_midi(self, input_fn, output_fn, smooth=0.2, minduration=0.1):
        # fs = 44100
        # hop = 128

        # audio = es.EqloudLoader(filename=input_fn, sampleRate=fs)()
        # start = self.segment(audio)
        # audio = audio[start:]
        # pitch_values, _ = self.pitch_extractor(audio)

        # try:
        #     bpm = self.rythm_extractor(audio)[0]
        # except:
        #     print(f"Skipping {input_fn} because of rythm extractor")
        #     return

        # pitch_values = np.insert(pitch_values, 0, values=[0]*8)

        # midi_pitch = self.hz2midi(pitch_values)
        # notes = self.midi_to_notes(midi_pitch, fs, hop, smooth, minduration)

        # self.save_midi(output_fn, notes, bpm)
        fs = 44100
        hop = 128
        audio = es.MonoLoader(filename=input_fn, downmix='mix', sampleRate=44100)()
        start = self.segment(audio)
        audio = audio[start:]
        # pitch_values,_ = self.pitch_extractor(audio)
        params = {"minfqr": 100.0, "maxfqr": 800.0, "voicing": 0.2, "minpeaksalience": 0.0}
        data = vamp.collect(audio, fs, "mtg-melodia:melodia", parameters=params)
        hop, pitch_values = data['vector']
        timestamps = 8 * 128/44100.0 + np.arange(len(pitch_values)) * (128 / 44100.0)
        pitch_values = np.array(pitch_values)


        bpm = self.rythm_extractor(audio)[0]

        pitch_values = np.insert(pitch_values, 0, values=[0]*8)

        midi_pitch = self.hz2midi(pitch_values)
        notes = self.midi_to_notes(midi_pitch, fs, 128, smooth, minduration)

        self.save_midi(output_fn, notes, bpm)


if __name__ == "__main__":
    # path = "data/2020/San Marino_Freaky!_Senhit.mp3"
    # output_fn = "midi_output/San Marino_Freaky!_Senhit.mid"

    proc = Preprocessor()

    # for year in range(2011, 2021):
    #     print(year)
    #     input_fns = os.listdir("data/"+str(year))
    #     output_fns = [name.split(".")[0] + ".mid" for name in input_fns]
    #     try:
    #         os.mkdir("midi_output/"+str(year))
    #     except FileExistsError:
    #         pass
    #     for idx in range(len(input_fns)):
    #         proc = Preprocessor()
    #         proc.audio_to_midi("data/"+str(year)+"/"+input_fns[idx], "midi_output/"+str(year)+"/"+output_fns[idx])
    
    # for idx in range(len(input_fns)):
    #     print(input_fns[idx])
    #     audio_to_midi("data/2020/" + input_fns[idx], "midi_output/" + output_fns[idx])

    # for idx in range(len(input_fns)):
    #     print(input_fns[idx])
    #     audio = es.EqloudLoader(filename="data/1956/" + input_fns[idx], sampleRate=44100)()
    #     segment(audio)

    
    # audio_to_midi(path, output_fn)


    # print(len(audio) / sr)

    # pitch_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)
    # pitch_values, pitch_confidence = pitch_extractor(audio)

    # pitch_times = np.linspace(0.0, len(audio) / sr, len(pitch_values))

    # # %%
    # fig = px.line(x=pitch_times, y=pitch_values)
    # fig.show()
    filename = "data/2020/Iceland_Think About Things_Dai & Gagnamagni.mp3"
    output = "test.mid"

    proc.audio_to_midi(filename, output)


# # f, axarr = plt.subplots(2, sharex=True)
# # axarr[0].plot(pitch_times, pitch_values)
# # axarr[0].set_title('estimated pitch [Hz]')
# # axarr[1].plot(pitch_times, pitch_confidence)
# # axarr[1].set_title('pitch confidence')
# # plt.show()
