import numpy as np
import cv2

class IoU:
    pred_box = []
    gt_box = []

    LAST_SHAPE = None

    def __init__(self, ls):
        self.LAST_SHAPE = ls
        pass

    def calculate(self, a, b):
        # print(f"{a} === {b}")
        self.pred_box.append(a)
        self.gt_box.append(b)

        res = -1. # impossible IoU
        if len(self.pred_box) == self.LAST_SHAPE:

            ixmin = np.maximum(self.pred_box[0], self.gt_box[0])
            ixmax = np.minimum(
                self.pred_box[0] + self.pred_box[2], self.gt_box[0] + self.gt_box[2]
            )
            iymin = np.maximum(self.pred_box[1], self.gt_box[1])
            iymax = np.minimum(
                self.pred_box[1] + self.pred_box[3], self.gt_box[1] + self.gt_box[3]
            )

            # if no overlapse, iou = 0
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)

            # area of intersection
            inters = np.multiply(iw, ih)

            # area of union
            uni = np.subtract(
                np.add(
                    np.multiply(self.pred_box[2], self.pred_box[3]),
                    np.multiply(self.gt_box[2], self.gt_box[3]),
                ),
                inters,
            )

            res = np.divide(inters, uni)

            self.pred_box = []
            self.gt_box = []

        return res

# x, y, width, height -> 4 features
raw_iou = np.frompyfunc(IoU(4).calculate, 2, 1)

def calculate_iou(pred_box, gt_box):
    pred_box = np.ascontiguousarray(pred_box)
    gt_box = np.ascontiguousarray(gt_box)

    # pred_box = np.asfortranarray(pred_box)
    # gt_box = np.asfortranarray(gt_box)

    iou = np.max(raw_iou(pred_box, gt_box), axis=len(pred_box.shape) - 1)

    return iou

# produce more than 2000 samples
def selective_search(img):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()  # fast mode
    boxes = ss.process()

    return boxes

def pair_bboxs_overlapse(ss_res, gt, filter_rate=0.5):
    if len(ss_res.shape) <= 1: ss_res = np.expand_dims(ss_res, 0)
    if len(gt.shape) <= 1: gt = np.expand_dims(gt, 0)

    expand_ss = np.expand_dims(ss_res, 0)
    expand_gt = np.expand_dims(gt, 1)

    iou_per_gt = calculate_iou(expand_ss, expand_gt)

    filter_iou_gt = np.where(iou_per_gt >= filter_rate, 1, 0)
    gt_count = np.sum(filter_iou_gt, axis=1)

    gt_ids = np.repeat(
        np.arange(gt.shape[0]),
        gt_count
    )

    filter_iou_ss = np.where(iou_per_gt >= filter_rate, True, False)

    tile_ss = np.tile(expand_ss, (iou_per_gt.shape[0], 1, 1))
    filter_ss = tile_ss[filter_iou_ss]

    return np.concatenate(
        (np.expand_dims(filter_ss, axis=1), np.expand_dims(gt[gt_ids], axis=1)), axis=1
    )

def pair_bboxs_max(ss_res, gt, filter_rate=0.5):
    if len(ss_res.shape) <= 1: ss_res = np.expand_dims(ss_res, 0)
    if len(gt.shape) <= 1: gt = np.expand_dims(gt, 0)

    expand_ss = np.expand_dims(ss_res, 0)
    expand_gt = np.expand_dims(gt, 1)

    iou_per_gt = calculate_iou(expand_ss, expand_gt)

    print(iou_per_gt)
    max_iou_each_region = np.max(iou_per_gt, axis=0)
    max_iou_region_id = np.argmax(iou_per_gt, axis=0)

    filter_idx = np.where(max_iou_each_region >= np.float32(filter_rate))

    if ss_res[filter_idx].shape[0] <= 0:
        return np.array([])

    return np.concatenate((
        np.expand_dims(ss_res[filter_idx], axis=1),
        np.expand_dims(gt[max_iou_region_id[filter_idx]], axis=1)
    ), axis=1) 
