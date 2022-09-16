import numpy as np
from .optimization import OCOpt
from .Annotations import BBox, predBBox, Annotations
import pulp


class OC_Cost:
    def __init__(self, lm=1, iou_mode=False):
        self.lm = lm
        if not iou_mode:
            self.mode = "giou"
        if iou_mode:
            self.mode = "iou"

    def getIntersectUnion(self, truth: BBox, pred: predBBox):
        a_area = (truth.width + 1) * (truth.height + 1)
        b_area = (pred.width + 1) * (pred.height + 1)

        w = max(0, self.__min_x2 - self.__max_x1 + 1)
        h = max(0, self.__min_y2 - self.__max_y1 + 1)
        intersect = w * h

        union = a_area + b_area - intersect

        return intersect, union

    def getIOU(self, truth: BBox, pred: predBBox):

        intersect, union = self.getIntersectUnion(truth, pred)

        iou = intersect / (union)
        return iou

    def getGIOU(self, truth: BBox, pred: predBBox):
        c_area = ((self.__max_x2 - self.__min_x1 + 1) *
                  (self.__max_y2 - self.__min_y1 + 1))

        intersect, union = self.getIntersectUnion(truth, pred)

        iou = intersect / (union)
        Giou = iou - ((c_area - union) / c_area)

        return Giou

    def getCloc(self, truth: BBox, pred: predBBox):
        """get Cloc

        Args:
            truth (dict): set truth bbox dict
            pred (dict): set pred bbox dict including precision
            mode (str): set mode of union cal

        Returns:
            float: C_loc
        """
        cost: float = 0

        # pre-calculate min max
        if truth.x > pred.x:
            self.__min_x1 = pred.x
            self.__max_x1 = truth.x
        else:
            self.__min_x1 = truth.x
            self.__max_x1 = pred.x
        if truth.y > pred.y:
            self.__min_y1 = pred.y
            self.__max_y1 = truth.y
        else:
            self.__min_y1 = truth.y
            self.__max_y1 = pred.y
        if truth.x2 > pred.x2:
            self.__min_x2 = pred.x2
            self.__max_x2 = truth.x2
        else:
            self.__min_x2 = truth.x2
            self.__max_x2 = pred.x2
        if truth.y2 > pred.y2:
            self.__min_y2 = pred.y2
            self.__max_y2 = truth.y2
        else:
            self.__min_y2 = truth.y2
            self.__max_y2 = pred.y2

        if self.mode == "giou":
            cost = (1 - self.getGIOU(truth, pred)) / 2
        if self.mode == "iou":
            cost = (1 - self.getIOU(truth, pred)) / 2
        return cost

    def getCcls(self, truth: BBox, pred: predBBox):
        """get Ccls

        Args:
            truth (BBox): set truth bbox dict
            pred (predBBox): set pred bbox dict including precision

        Returns:
            float: C_cls
        """
        clt = truth.label
        clp = pred.label

        preci = pred.precision
        ccls = 0.5
        if clt == clp:
            ccls = (1 - preci) / 2
        else:
            ccls = (1 + preci) / 2
        return ccls

    def getoneCost(self, truth, pred):
        """get C_ij cost

        Args:
            truth (dict): set truth bbox dict
            pred (dict): set pred bbox dict including precision

        Returns:
            float: C_ij cost
        """
        Cloc = self.getCloc(truth, pred)
        CCls = self.getCcls(truth, pred)

        return (self.lm * Cloc) + ((1 - self.lm) * CCls)

    def build_C_matrix(self, truth_annotations: Annotations, pred_annotations: Annotations):
        n = len(truth_annotations.bboxs)
        m = len(pred_annotations.bboxs)

        self.cost = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                self.cost[i][j] = self.getoneCost(
                    truth_annotations.bboxs[j], pred_annotations.bboxs[i])
        return self.cost

    def optim(self, beta):
        m = self.cost.shape[0] + 1
        n = self.cost.shape[1] + 1
        opt = OCOpt(m, n, beta)
        opt.set_cost_matrix(self.cost)
        opt.setVariable()
        opt.setObjective()
        opt.setConstrain()

        result = opt.prob.solve(pulp.PULP_CBC_CMD(
            msg=0, timeLimit=100))
        p_matrix = np.zeros((m, n))

        # print('objective value: {}'.format(pulp.value(opt.prob.objective)))
        # print('solution')
        for i in range(opt.m):
            for j in range(opt.n):
                #print(f'{opt.variable[j][i]} = {pulp.value(opt.variable[j][i])}')
                p_matrix[j][i] = pulp.value(opt.variable[j][i])
        p_matrix[-1][-1] = 0
        p_tilde_matrix = p_matrix / np.sum(p_matrix)
        self.p_matrix = p_matrix
        self.p_tilde_matrix = p_tilde_matrix
        self.opt = opt
        return p_tilde_matrix
