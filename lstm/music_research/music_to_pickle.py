import os 
from mido import MidiFile, Message, MidiTrack
import pickle
midi_files_list =  os.listdir('../../midi_output/') 
import numpy as np

notes = []
for year in range(1956, 2021):
    print(year)
    midi_files_list =  os.listdir('../../midi_output/'+str(year)) 
    for file in midi_files_list:
        midi = MidiFile(filename='../../midi_output/'+str(year)+"/"+file)
        # print(midi.print_tracks())
        time = float(0)
        prev = float(0)

        for msg in midi:
            time += msg.time

            if not msg.is_meta:
                # if msg.channel in [1,2,3]:
                    if msg.type == 'note_on':
                        note = msg.bytes()
                        note = note[1:3]
                        note.append(time - prev)
                        prev = time
                        notes.append(note)

# notes = np.array(notes)

with open('./notes2.pkl','wb') as f:
    pickle.dump(notes, f, protocol=2)