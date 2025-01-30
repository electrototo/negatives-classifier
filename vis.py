"""
References
* https://github.com/pjreddie/darknet/blob/master/src/image.c#L293 (BBX coordinates)
"""

import argparse
import cv2
import json


ap = argparse.ArgumentParser()
ap.add_argument('-c', '--classification', required=True, help='path to classification file')

args = vars(ap.parse_args())


classification = []
with open(args['classification'], 'r') as f:
    for line in f.readlines():
        classification.append(json.loads(line))


# Filter out images that were not classified
data = list(filter(lambda x: len(x['classification']) > 0, classification))

# For each file open it, and print the classification
for entry in data:
    image = cv2.imread(entry['file'])

    # Save the original image with bounding boxes
    for object in entry['classification']:
        print('[object]', object[0], object[1])

        coords = object[2]

        x = int(coords[0] - (coords[2] / 2))
        y = int(coords[1] - (coords[3] / 2))

        image = cv2.rectangle(image, (x, y), (x + int(coords[2]), y + int(coords[3])), (0, 255, 0), 5)

    # width / height
    w = image.shape[1]
    h = image.shape[0]

    ratio = w / h

    # new width / new height
    nw = 640
    nh = int(480 * ratio)

    resized = cv2.resize(image, (nh, nw), None)

    cv2.imshow('image', resized)
    key = cv2.waitKey(0)

    print()

    if key == ord('q'):
        break

cv2.destroyAllWindows()

# y/x