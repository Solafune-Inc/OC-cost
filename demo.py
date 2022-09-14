import numpy as np
from oc_cost.Annotations import Annotation_images, Prediction_images
from oc_cost.oc_cost import OC_Cost
from tqdm import tqdm
import json
import argparse
import math

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='calculate OC cost')

    parser.add_argument(
        '-pd', '--pred', help='pred json')
    parser.add_argument(
        '-gt', '--truth', help='ground truth json')
    parser.add_argument(
        '-lm', '--lam', help='Lambda parameter', default=1
    )
    parser.add_argument(
        '-b', '--beta', help='beta parameter', default=0.5
    )
    parser.add_argument(
        '--iou_mode', help="turn on iou mode", action='store_true'
    )

    args = parser.parse_args()

    pred_path = args.pred
    truth_path = args.truth

    preds: Prediction_images = Prediction_images()
    truth: Annotation_images = Annotation_images()
    occost = OC_Cost(float(args.lam), args.iou_mode)
    total_occost = 0
    with open(pred_path) as f:
        pd_dict = json.load(f)
        preds.load_from_dict(pd_dict)

    with open(truth_path) as f:
        gt_dict = json.load(f)
        truth.load_from_dict(gt_dict)

    for image_name in tqdm(truth.keys()):
        c_matrix = occost.build_C_matrix(truth[image_name], preds[image_name])
        pi_tilde_matrix = occost.optim(float(args.beta))
        cost = np.sum(np.multiply(pi_tilde_matrix, occost.opt.cost))
        if math.isnan(cost):
            cost = 0
        total_occost += cost

    oc_cost = total_occost / len(truth.keys())
    print(oc_cost)
