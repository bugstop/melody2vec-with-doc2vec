import gensim
import json

from flask import Flask, request, jsonify, make_response, render_template
from flask_cors import CORS

app = Flask(__name__, )
CORS(app, resources=r'/*')
DEBUG = True

with open('melody.json') as f:
    melody = json.load(f)

model = gensim.models.Doc2Vec.load('d2v.bin')

# inference hyper-parameters
alpha = 0.01
epoch = 1000


def sims(filename, infer=False, positive=True, n=10):
    for song in melody:
        if filename == song[-1]:
            idx = song[0]
            break
    else:
        return []

    if infer:
        words = model.infer_vector(melody[idx][1].split(' '), alpha=alpha, steps=epoch)
    else:
        words = model.docvecs[idx]

    if positive:
        neighbors = model.docvecs.most_similar(positive=[words], topn=n)
    else:
        neighbors = model.docvecs.most_similar(negative=[words], topn=n)
        neighbors = neighbors[::-1]

    neighbors = [(idx+1, str(item[0]), melody[item[0]][-1], str(item[-1] * 100)[:5] + '%') for (idx, item) in enumerate(neighbors)]

    return neighbors

@app.route('/', methods=['GET'])
def index():
    filename = request.args.get('filename')
    if not filename:
        return render_template('base.html', title='音乐搜索', results=[])
    if filename.isdigit():
        filename = melody[int(filename)][-1]

    infer = request.args.get('infer')
    if infer == 'true':
        infer = True
    else:
        infer = False

    negative = request.args.get('negative')
    if negative == 'true':
        negative = True
    else:
        negative = False

    n = request.args.get('n')
    if not n:
        n = 10
    else:
        n = int(n)

    x = sims(filename, infer=infer, positive=not negative, n=n)
    return render_template('base.html', title='搜索结果', results=x, filename=filename)

def construct_response(msg) :
    """完成请求响应的构造"""
    response = make_response(jsonify(msg))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,HEAD,GET,POST'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with'
    return response

if __name__ == "__main__":
    app.run(port=80)
