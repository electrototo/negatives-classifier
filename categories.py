"""
Will retrieve all categories and go through the pictures containing the selected category
"""

import json
import argparse
import cv2
import numpy as np

from PIL import Image, ImageFont, ImageDraw


ap = argparse.ArgumentParser()
ap.add_argument('-s', '--skip', help='I', action='store_true')

args = vars(ap.parse_args())


def create_categories_and_data(save=False):
    categories = {}
    data = {}

    with open('classification.txt', 'r') as f:
        for line in f.readlines():
            parsed = json.loads(line)

            # Assuming that each file contained has an unique name...
            data[parsed['file']] = parsed['classification']

            for entry in parsed['classification']:
                label = entry[0]

                if label not in categories:
                    categories[label] = []

                categories[label].append(parsed['file'])

    if save:
        with open('categories.txt', 'w') as f:
            json.dump(categories, f)

        with open('index_data.txt', 'w') as f:
            json.dump(data, f)

    return categories, data


def load_categories_and_data():
    with open('categories.txt', 'r') as f:
        categories = json.load(f)

    with open('index_data.txt', 'r') as f:
        data = json.load(f)

    return categories, data


if not args['skip']:
    categories, data = create_categories_and_data(save=True)
else:
    categories, data = load_categories_and_data()

cat_count = [(category, len(entries)) for category, entries in categories.items()]
cat_count.sort(reverse=True, key=lambda x: x[1])

for index, (category, count) in enumerate(cat_count):
    print('[{}] {} ({})'.format(index, category, count))

selection = int(input('> '))
selected_category = cat_count[selection][0]

entries = categories[selected_category]

current_idx = 0

# ==== font sheningans ====

font_path = '/usr/share/fonts/truetype/ubuntu/Ubuntu-Th.ttf'
font_size = 24
padding = 4

font = ImageFont.truetype(font_path, font_size)
(left, top, right, bottom) = font.getbbox(selected_category)

# =========

# === controls ===
show_bbx = False
# === controls ===


while True:
    filename = entries[current_idx]
    current_data = data[filename]

    image = cv2.imread(filename)

    w, h = image.shape[1], image.shape[0]
    nw, nh = w // 6, h // 6

    ratio = w / h

    image = cv2.resize(image, (nw, nh), None)

    image_pil = Image.fromarray(image)

    draw = ImageDraw.Draw(image_pil)
    draw.rectangle([(0, nh - bottom - top - padding), (right + padding * 2, nh)], fill='#201810')
    draw.text((0 + padding, nh - bottom - top - padding), selected_category, font=font, fill='#15e7fe')

    image = np.array(image_pil)

    if show_bbx:
        bounding_boxes = list(map(lambda x: x[2], filter(lambda x: x[0] == selected_category, current_data)))

        for coords in bounding_boxes:
            bbx_left = int(coords[0] - (coords[2] / 2)) // 6
            bbx_top = int(coords[1] - (coords[3] / 2)) // 6
            bbx_right = bbx_left + int(coords[2] / 6)
            bbx_bottom = bbx_top + int(coords[3] / 6)

            image = cv2.rectangle(image, (bbx_left, bbx_top), (bbx_right, bbx_bottom), (0, 255, 0), 2)


    cv2.imshow('display', image)
    key = cv2.waitKey(0)

    if key == ord('q'):
        break
    elif key == ord('a'):
        current_idx = max(current_idx - 1, 0)

    elif key == ord('d'):
        current_idx = min(current_idx + 1, len(entries) - 1)

    elif key == ord('w'):
        current_idx = 0

        selection = max(selection - 1, 0)
        selected_category = cat_count[selection][0]
        entries = categories[selected_category]

        (left, top, right, bottom) = font.getbbox(selected_category)

    elif key == ord('s'):
        current_idx = 0

        selection = min(selection + 1, len(cat_count) - 1)
        selected_category = cat_count[selection][0]
        entries = categories[selected_category]

        (left, top, right, bottom) = font.getbbox(selected_category)

    elif key == ord('m'):
        show_bbx = not show_bbx


cv2.destroyAllWindows()