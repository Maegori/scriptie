import plotly.express as px
import essentia
import essentia.standard  as es
import numpy as np 
from midiutil.MidiFile import MIDIFile
import os

from scipy.signal import medfilt

def save_midi(output_fn, notes, tempo):

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

def midi_to_notes(midi, fs, hop, smooth, minduration):

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


def hz2midi(hz):

    # convert from Hz to midi note
    hz_nonneg = hz.copy()
    idx = hz_nonneg <= 0
    hz_nonneg[idx] = 1
    midi = 69 + 12*np.log2(hz_nonneg/440.)
    midi[idx] = 0

    # round
    midi = np.round(midi)

    return midi


def audio_to_midi(input_fn, output_fn, smooth=0.25, minduration=0.1):
    fs = 44100
    hop = 128

    audio = es.EqloudLoader(filename=input_fn, sampleRate=fs)()
    pitch_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)
    pitch_values, pitch_confidence = pitch_extractor(audio)

    tempo = es.RhythmExtractor2013()(audio)

    bpm = tempo[0]

    pitch_values = np.insert(pitch_values, 0, values=[0]*8)

    midi_pitch = hz2midi(pitch_values)
    notes = midi_to_notes(midi_pitch, fs, hop, smooth, minduration)

    save_midi(output_fn, notes, bpm)


if __name__ == "__main__":
    # path = "../data/2020/San Marino_Freaky!_Senhit.mp3"
    # output_fn = "midi_output/San Marino_Freaky!_Senhit.mid"

    # input_fns = os.listdir("../data/2020")
    # output_fns = [name.split(".")[0] + ".mid" for name in input_fns]
    
    # for idx in range(len(input_fns)):
    #     print(input_fns[idx])
    #     audio_to_midi("../data/2020/" + input_fns[idx], "midi_output/" + output_fns[idx])

    sr = 44100

    audio = es.EqloudLoader(filename="../data/2020/Iceland_Think About Things_Daði & Gagnamagnið.mp3", sampleRate=44100)()
    # audio_to_midi(path, output_fn)


    print(len(audio) / sr)

    pitch_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)
    pitch_values, pitch_confidence = pitch_extractor(audio)

    pitch_times = np.linspace(0.0, len(audio) / sr, len(pitch_values))

    # %%
    fig = px.line(x=pitch_times, y=pitch_values)
    fig.show()

# # f, axarr = plt.subplots(2, sharex=True)
# # axarr[0].plot(pitch_times, pitch_values)
# # axarr[0].set_title('estimated pitch [Hz]')
# # axarr[1].plot(pitch_times, pitch_confidence)
# # axarr[1].set_title('pitch confidence')
# # plt.show()