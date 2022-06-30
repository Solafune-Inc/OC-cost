
from oc_cost import OC_Cost
from Annotations import Annotation_images, Prediction_images
import json
from tqdm import tqdm
import numpy as np


def test_basic_oc_cost():

    pred_path = "expjson/pred.json"
    truth_path = "expjson/truth.json"
    lam = 1
    beta = 0.5

    lam = 1
    occost = OC_Cost(float(lam), False)
    preds: Prediction_images = Prediction_images()
    truth: Annotation_images = Annotation_images()
    total_occost = 0
    with open(pred_path) as f:
        pd_dict = json.load(f)
        preds.load_from_dict(pd_dict)

    with open(truth_path) as f:
        gt_dict = json.load(f)
        truth.load_from_dict(gt_dict)

    for image_name in tqdm(truth.keys()):
        occost.build_C_matrix(truth[image_name], preds[image_name])
        pi_tilda_matrix = occost.optim(float(beta))

        total_occost += np.sum(np.multiply(pi_tilda_matrix, occost.opt.cost))
    oc_cost = total_occost / len(truth.keys())
    assert oc_cost == 0.08928571428571429


def test_basic_no_cost():
    pred_path = "expjson/truth.json"
    truth_path = "expjson/truth.json"

    beta = 0.5

    preds: Prediction_images = Prediction_images()
    truth: Annotation_images = Annotation_images()

    lam = 1
    occost = OC_Cost(float(lam), False)
    total_occost = 0
    with open(pred_path) as f:
        pd_dict = json.load(f)
        preds.load_from_dict(pd_dict)

    with open(truth_path) as f:
        gt_dict = json.load(f)
        truth.load_from_dict(gt_dict)

    for image_name in tqdm(truth.keys()):
        cost = occost.build_C_matrix(truth[image_name], preds[image_name])

        pi_tilda_matrix = occost.optim(float(beta))
        print(pi_tilda_matrix)
        #print(occost.opt.cost)
        total_occost += np.sum(np.multiply(pi_tilda_matrix, occost.opt.cost))

    oc_cost = total_occost / len(truth.keys())
    assert oc_cost == 0
