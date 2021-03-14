import os
import json

path = "./melody-segmented/"

files = os.listdir(path)

melody = []

for index, filename in enumerate(files):
    with open(path + filename) as f:
        text = f.read().replace('\n', ' ')
    melody.append([index, text, filename.split('_')[0] + '.mid'])

print(melody)

with open("melody.json", "w") as f:
    json.dump(melody, f)
