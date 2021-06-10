import os
import json
import numpy as np

import settings


def cosine_similarity(x, y, norm=False):
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos


def midi(file_name):
    beat_clock = settings.beat_clock
    time_code = settings.time_code
    velocity = settings.velocity
    binary = settings.binary
    key = settings.key

    midi_header = b'MThd'
    midi_header_length = b'\x00\x00\x00\x06'
    single_multi_channel_track = b'\x00\x01'
    number_of_tracks = b'\x00\x01'
    time_code_based_time = b'\x00\x80'  # 0x0080 means: 512 beats for a whole note.
    track_header = b'MTrk'
    track_header_length = b'\x00\x00\x00\x00'  # Total length.
    start = midi_header + midi_header_length + single_multi_channel_track + number_of_tracks + time_code_based_time + track_header
    body = b'\x00\xc0\x00'
    stop = b'\x81\x00\xff\x2f\x00'

    tempo = 1

    def generator(time_start, note_1, note_2, time_last):
        def get_time_code(t):
            if t[0] == 't':
                value = t[1:]
            else:
                value = beat_clock[t]
            rc = time_code[str(int(int(value) / tempo))]
            return rc

        note = str(int(note_2) * 12 + key[note_1] + 24)
        line_start = get_time_code(time_start) + velocity['1'][0] + \
                     binary[note] + velocity['1'][1]
        line_stop = get_time_code(time_last) + velocity['0'][0] + \
                    binary[note] + velocity['0'][1]
        return line_start + line_stop

    with open(f'{file_name}.txt') as f_obj:
        notes = f_obj.read().split('\n')

    for note_ in notes:
        if not note_ or note_[0] == '#':
            continue
        if note_[:5] == 'tempo':
            tempo = float(note_.split('=')[-1])
            continue
        _a, _b, _c, _d = note_.split('.')
        body += generator(_a, _b, _c, _d)
    body += stop

    length = len(body)
    length, _d = divmod(length, 256)
    length, _c = divmod(length, 256)
    length, _b = divmod(length, 256)
    length, _a = divmod(length, 256)
    if length:
        print('Too Long!')
        exit(1)
    track_header_length = binary[str(_a)] + binary[str(_b)] + binary[str(_c)] + binary[str(_d)]
    file = start + track_header_length + body

    with open(f'{file_name}.mid', 'wb') as f_obj:
        f_obj.write(file)


key_map = {
    'C': '1',
    'C#': '#1',
    'Db': 'b2',
    'D': '2',
    'D#': '#2',
    'Eb': 'b3',
    'E': '3',
    'F': '4',
    'F#': '#4',
    'Gb': 'b5',
    'G': '5',
    'G#': '#5',
    'Ab': 'b6',
    'A': '6',
    'A#': '#6',
    'Bb': 'b7',
    'B': '7',
}

with open('melody.json') as f:
    melody = json.load(f)

index = int(input(f'midi file index 0-{len(melody) - 1}: '))

melody_sample_orig = melody[index][1].split(' ')

for i in range(len(melody_sample_orig)):
    print(i, melody_sample_orig[i])
edit_melody = int(input(' -> '))

if not os.path.isdir(f'melody_{index}_{edit_melody}'):
    os.mkdir(f'melody_{index}_{edit_melody}')
with open(f'melody_{index}_{edit_melody}/sample_orig.txt', 'w') as f:
    f.write('\n'.join(melody_sample_orig))

with open('w2v.json') as f:
    words = json.load(f)

melody = melody_sample_orig[edit_melody]
ref = 0
while words[ref][0] != melody:
    ref += 1


def melody_time(m):
    m = m.replace('C', ' C').replace('D', ' D').replace('E', ' E') \
        .replace('F', ' F').replace('G', ' G').replace('A', ' A') \
        .replace('B', ' B').replace('R', ' R').replace('  ', ' ')
    m = m.split(' ')[1:]
    t = [float(n.split('-')[-1]) for n in m]
    return sum(t)


print(ref, melody, melody_time(melody))

for i in range(len(words)):
    words[i].append(cosine_similarity(words[ref][1], words[i][1]))

words.sort(reverse=True, key=lambda z: z[-1])

target_length = melody_time(melody)

words_cand = words[1:] + [words[0]]

for word in words:
    candidate = word[0]
    candidate_length = melody_time(candidate)

    if abs(target_length - candidate_length) > 0.5:
        print((candidate, candidate_length, word[-1]), 'x')
        continue

    if input((candidate, candidate_length, word[-1], '->')):
        break

    melody_sample = melody_sample_orig[:]
    melody_sample[edit_melody] = word[0]
    melody_sample = ''.join(melody_sample)

    melody_sample = melody_sample.replace('C', ' C').replace('D', ' D').replace('E', ' E') \
        .replace('F', ' F').replace('G', ' G').replace('A', ' A') \
        .replace('B', ' B').replace('R', ' R').replace('  ', ' ')
    melody_sample = melody_sample.split(' ')[1:]


    def note_transform(note_orig):
        peach, length = note_orig.split('-')

        if peach.startswith('R'):
            note_1, note_2 = 'R', 'R'
        else:
            note_1, note_2 = key_map[peach[:-1]], peach[-1]

        time_last = int(4 / float(length) + 0.5)
        return '0', str(note_1), note_2, str(time_last)


    melody_sample = list(map(note_transform, melody_sample))

    sample = ''
    r_length = 2
    while melody_sample:
        note = melody_sample.pop(0)
        if note[1] == 'R':
            r_length = note[-1] if not r_length else int(4 / (4 / r_length + 4 / int(note[-1])) + 0.4)
        else:
            sample += f'{r_length}.{note[1]}.{note[2]}.{note[-1]}\n'
            r_length = 0

    with open(f'melody_{index}_{edit_melody}/sample_{word[-1]:.2g}_{word[0]}.txt', 'w') as f:
        f.write(sample)

    midi(f'melody_{index}_{edit_melody}/sample_{word[-1]:.2g}_{word[0]}')

log = [('file index', index), ('melody word index', edit_melody)] + \
      [(word[0], word[-1]) for word in words[:5]]
with open(f'melody_{index}_{edit_melody}/sample_log.json', 'w') as f:
    json.dump(log, f)
