# OC-cost (Optimal Correction Cost) with pulp

This repository is an implementation of oc-cost, implemented by pulp.

Mayu Otani, Riku Togashi, Yuta Nakashima, Esa Rahtu, Janne Heikkilä, Shin'ichi Satoh(2022). Optimal Correction Cost for Object Detection Evaluation
https://arxiv.org/abs/2203.14438

```
@InProceedings{Otani_2022_CVPR,
    author    = {Otani, Mayu and Togashi, Riku and Nakashima, Yuta and Rahtu, Esa and Heikkil\"a, Janne and Satoh, Shin'ichi},
    title     = {Optimal Correction Cost for Object Detection Evaluation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {21107-21115}
}
```

## How to install
from github
```
git clone https://github.com/Solafune-Inc/OC-cost.git
pip install .
```


## Usage
### OC_Cost Calculation
```python
from oc_cost import OC_Cost
from oc_cost.Annotations import Annotation_images, Prediction_images

import json
import numpy as np

if __name__ == "__main__":
    pred_path = "./expjson/pred.json"
    truth_path = "./expjson/truth.json"
    lam = 1
    beta = 0.5

    preds: Prediction_images = Prediction_images()
    truth: Annotation_images = Annotation_images()
    occost = OC_Cost(float(lam), False)
    total_occost = 0
    with open(pred_path) as f:
        pd_dict = json.load(f)
        preds.load_from_dict(pd_dict)

    with open(truth_path) as f:
        gt_dict = json.load(f)
        truth.load_from_dict(gt_dict)

    for image_name in tqdm(truth.keys()):
        c_matrix = occost.build_C_matrix(truth[image_name], preds[image_name])
        pi_tilde_matrix = occost.optim(float(beta))
        cost = np.sum(np.multiply(pi_tilde_matrix, occost.opt.cost))
        total_occost += cost

    oc_cost = total_occost / len(truth.keys())
    print(oc_cost)
```


### Traslate coco format to this format
if you translate ground truth file
```bash
python coco2Annotations.py -i ./input.json -o ./output.json
```

if you translate predictions file
```bash
python coco2Annotations.py -i ./input.json -o ./output.json --prediction
```