import os

classes = ['airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle']


def getObjects(FilePath):
    objects = []
    with open(FilePath) as f:
        for line in f.readlines():
            lineArray = line.strip().split(',')
            xmin_s = lineArray[0].strip('(').strip(')').replace(' ','')
            ymin_s = lineArray[1].strip('(').strip(')').replace(' ','')
            xmax_s = lineArray[2].strip('(').strip(')').replace(' ','')
            ymax_s = lineArray[3].strip('(').strip(')').replace(' ','')
            xmin = float(xmin_s)
            ymin = float(ymin_s)
            xmax = float(xmax_s)
            ymax = float(ymax_s)
            assert xmax >= xmin
            assert ymax >= ymin
            label = classes[int(lineArray[4])-1]
            # diff = lineArray[5]   ---->NWPU_VHR datasets has no diff label
            diff = 0
            objects.append({
                "wnid": label, # class name, not number
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
