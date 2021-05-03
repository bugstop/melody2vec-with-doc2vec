import json
from shutil import copyfile

with open('melody.json') as f:
    melody = json.load(f)

for index in [0, 245, 9533]:
    filename = melody[index][-1]
    input('copy ' + filename + ' -> ')
    midi = './midi/' + f"{filename[0]}/"
    copyfile(midi + filename, f'z{index:05d}_' + filename)
