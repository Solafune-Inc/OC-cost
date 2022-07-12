import json
from collections import defaultdict
import argparse


def coco2Annotations(coco_dict: dict):
    annotations_dict = {
        "images": list()
    }
    images_annotations = defaultdict(list)

    image_pair = dict()
    category_pair = dict()
    for image in coco_dict["images"]:
        image_pair[(image["id"])] = image["file_name"]

    for category in coco_dict["categories"]:
        category_pair[(category["id"])] = category["name"]

    for bbox in coco_dict["annotations"]:
        image_name = image_pair[bbox["image_id"]]
        category_name = category_pair[bbox["category_id"]]
        lefttop_x = bbox["bbox"][0]
        lefttop_y = bbox["bbox"][1]
        width = bbox["bbox"][2]
        height = bbox["bbox"][3]
        images_annotations[image_name].append(
            {
                "class": category_name,
                "lefttop_x": lefttop_x,
                "lefttop_y": lefttop_y,
                "rightbottom_x": lefttop_x + width,
                "rightbottom_y": lefttop_y + height
            }
        )
    for key in images_annotations.keys():
        annotations_dict["images"].append(
            {
                "name": key,
                "annotation": images_annotations[key]
            }
        )

    return annotations_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='translate coco format to Annotations format')
    parser.add_argument(
        '-i', '--input', help='input json')
    parser.add_argument(
        '-o', '--output', help='output json')
    parser.add_argument('--prediction', help='is json file is prediction file',
                        action='store_false')

    args = parser.parse_args()

    load_dict = dict()

    with open(args.input) as f:
        load_dict = json.load(f)

    trans_dict = coco2Annotations(load_dict)

    with open(args.output, 'w') as fp:
        json.dump(trans_dict, fp)
