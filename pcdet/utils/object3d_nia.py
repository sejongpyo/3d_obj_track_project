import numpy as np
import json
import copy

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = json.load(f)
    objects = [Object3d(line) for line in lines['annotations']]
    return objects


def cls_type_to_id(cls_type):
    type_to_id = {'vehicle': 1, 'pedestrian': 2, 'bike': 3}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        label = line['attribute']
        self.src = line
        self.cls_type = line['class']
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(1)
        self.occlusion = float(1)
        self.box3d = np.array([float(label['location'][0]), float(label['location'][1]), float(label['location'][2]),
                               float(label['dimension'][0]), float(label['dimension'][1]), float(label['dimension'][2]), # lwh
                               float(label['yaw'])], dtype=np.float32)
        self.h = float(label['dimension'][2])
        self.w = float(label['dimension'][1])
        self.l = float(label['dimension'][0])
        self.loc = np.array((float(label['location'][0]), float(label['location'][1]), float(label['location'][2])), dtype=np.float32)
        self.ry = float(label['yaw'])
        self.alpha = float(1)
        self.box2d = np.array((float(1), float(1), float(1), float(1)), dtype=np.float32)
        self.score = float(0.0)
        self.level_str = 'Moderate'
        self.level = 1
        self.track_id = label['track_id']