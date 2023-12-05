import numpy as np


class BoundingBox(list):

    def __init__(self, bbox_list):
        """
        Bounding box.

        :param bbox_list: list. [sy, sx, ey, ex].
        """
        super().__init__(bbox_list)
        self.sy, self.sx, self.ey, self.ex = bbox_list

    def contains(self, bbox):
        if self.sy <= bbox[0] and self.sx <= bbox[1] and self.ey >= bbox[2] and self.ex >= bbox[3]:
            return True
        else:
            return False
