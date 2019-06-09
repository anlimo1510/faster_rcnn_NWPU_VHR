import os

classes = ['airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle']


def getObjects(FilePath):
    objects = []
    with open(FilePath) as f:
        for line in f.readlines():
            lineArray = line.rstrip().split(' ')
            xmin = float(lineArray[0])
            ymin = float(lineArray[1])
            xmax = float(lineArray[2])
            ymax = float(lineArray[3])
            assert xmax >= xmin
            assert ymax >= ymin
            label = lineArray[4]
            diff = lineArray[5]
            objects.append({
                "wnid": label,
                "difficult": diff,
                "box": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                }
            })
    return objects


def parse(filepath):
    return getObjects(filepath) 
