"""
Useful references:
* https://github.com/pjreddie/darknet/issues/241 (Invalid type)
* https://pytorch.org/vision/main/models.html
* https://github.com/pjreddie/darknet/issues/791 (CUDA OOM issue)
* https://stackoverflow.com/questions/64885148/error-iplimage-does-not-name-a-type-when-trying-to-build-darknet-with-opencv (magic)
"""

import argparse
import darknet as dn
import os
import json
import time


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--directory', required=True)

args = vars(ap.parse_args())

dn.set_gpu(0)

# Network loading
net = dn.load_net(b'data/yolov3.cfg', b'data/yolov3.weights', 0)
meta = dn.load_meta(b'data/coco.data')

# Detection
for dirpath, dirnames, filenames in os.walk(args['directory']):
    print(dirpath, dirnames, filenames)

    for file in filenames:
        path = os.path.join(dirpath, file)

        print('[detecting]', path)
        now = time.time()
        response = dn.detect(net, meta, path.encode('utf-8'))
        print('[done] {}s'.format(time.time() - now))

        # Need to decode as utf-8 the output
        classes = []
        for entry in response:
            classes.append((entry[0].decode('utf-8'),) + entry[1:])

        # Save as a new json line
        with open('classification.txt', 'a+') as f:
            data = json.dumps({ "file": path, "classification": classes })
            f.write(f'{data}\n')