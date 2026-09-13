import json
import dataset as db
from functools import cmp_to_key
from scipy.spatial import distance
import sys
argv = sys.argv[1:]

path = './models/kmeans-model.json'

with open(path) as fp:
    data = json.load(fp)
    centroids = data['centroids']


def compute(vec):
    min = 1
    for c in centroids:
        dis = distance.cosine(c, vec)
        if (dis < min):
            min = dis
    return min


def scale(xs, distance=0.015, scale=2):
    arr = []
    for x in xs:
        if (compute(x) >= distance):
            x = [scale*i for i in x]
        arr.append(x)
    return arr


def rank(xs):
    xs.sort(key=cmp_to_key(lambda x, y: 1 if compute(x) > compute(y) else -1))
    return xs


def predict(id):
    data = db.getXY2(id)
    xs = data['x']
    pt = []
    for x in xs:
        pt.append(compute(x))
    pt.sort(reverse=True)
    print(pt)


if (argv[0] == 'predict'):
    predict(int(argv[1]))
