import argparse
import os
import json
import time
import torch

import sys


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--directory', required=True)

args = vars(ap.parse_args())

# Network loading
net = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Detection
for dirpath, dirnames, filenames in os.walk(args['directory']):
    print(dirpath, dirnames, filenames)

    for file in filenames:
        path = os.path.join(dirpath, file)

        print('[detecting]', path)
        now = time.time()

        try:
            response = net(path)
            print('[done] {}s\n'.format(time.time() - now))

            classes = json.loads(response.pandas().xyxy[0].to_json(orient='records'))

            # Save as a new json line
            with open('classification-2.txt', 'a+') as f:
                data = json.dumps({ "file": path, "classification": classes })
                f.write(f'{data}\n')

        except KeyboardInterrupt:
            raise

        except:
            print('[skipping]', path)
            continue
