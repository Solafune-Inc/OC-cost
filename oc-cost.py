class OC_Cost:
    def __init__(self, lm=0.5):
        self.lm = lm

    def getIOU(self, truth, pred):
        a_area = (truth["rightbottom_x"] - truth["lefttop_x"] + 1) * \
            (truth["rightbottom_y"] - truth["lefttop_y"] + 1)
        b_area = (pred["rightbottom_x"] - pred["lefttop_x"] + 1) * \
            (pred["rightbottom_y"] - pred["lefttop_y"] + 1)

        abx_mn = max(truth["lefttop_x"], pred["lefttop_x"])
        aby_mn = max(truth["lefttop_y"], pred["lefttop_y"])
        abx_mx = min(truth["rightbottom_x"], pred["rightbottom_x"])
        aby_mx = min(truth["rightbottom_y"], pred["rightbottom_y"])
        w = max(0, abx_mx - abx_mn + 1)
        h = max(0, aby_mx - aby_mn + 1)
        intersect = w * h

        iou = intersect / (a_area + b_area - intersect)
        return iou
