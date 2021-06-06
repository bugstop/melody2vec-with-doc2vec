import json
from shutil import copyfile

with open('melody.json') as f:
    melody = json.load(f)

while True:
    x = input('-> ')
    if not x:
        break

    index = int(x)
    filename = melody[index][-1]

    input('copy ' + filename + ' -> ')
    
    midi = './midi/' + f"{filename[0]}/"
    copyfile(midi + filename, f'positive_{index:05d}_' + filename)
