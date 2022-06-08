

class BBox:
    def __init__(self, label, lefttop, width, height):
        self.label = label
        self.x = lefttop[0]
        self.y = lefttop[1]
        self.width = width
        self.height = height

    def get_lefttop(self):
        return (self.x, self.y)

    def get_rightbottom(self):
        return (self.x + self.width, self.y + self.height)

    def get_rightbottom_x(self):
        return (self.x + self.width)

    def get_rightbottom_y(self):
        return (self.y + self.height)


class predBBox(BBox):
    def __init__(self, label, lefttop, width, height, precision):
        self.label = label
        self.x = lefttop[0]
        self.y = lefttop[1]
        self.width = width
        self.height = height
        self.precision = precision


class Annotations:
    def __init__(self, image_name, bboxs: list):
        self.image_name = image_name
        self.bboxs = bboxs


class Annotation_images(dict):
    def load_from_dict(self, dic):
        for image in dic["images"]:
            name = image["name"]
            bboxs = list()
            for bbox in image["annotation"]:
                cls = bbox["class"]
                x = bbox["lefttop_x"]
                y = bbox["lefttop_y"]
                width = bbox["rightbottom_x"] - bbox["lefttop_x"]
                height = bbox["rightbottom_y"] - bbox["lefttop_y"]
                ins = BBox(cls, (x, y), width, height)
                bboxs.append(ins)
            self[name] = Annotations(name, bboxs)


class Prediction_images(dict):

    def load_from_dict(self, dic):
        for image in dic["images"]:
            name = image["name"]
            bboxs = list()
            for bbox in image["annotation"]:
                cls = bbox["class"]
                precision = 1
                if "precision" in bbox.keys():
                    precision = bbox["precision"]
                x = bbox["lefttop_x"]
                y = bbox["lefttop_y"]
                width = bbox["rightbottom_x"] - bbox["lefttop_x"]
                height = bbox["rightbottom_y"] - bbox["lefttop_y"]
                ins = predBBox(cls, (x, y), width, height, precision)
                bboxs.append(ins)
            self[name] = Annotations(name, bboxs)
