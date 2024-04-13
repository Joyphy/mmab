import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from skimage import measure

from mmab.registry import METRICS

from mmengine.logging import print_log
from mmengine.evaluator import BaseMetric

@METRICS.register_module()
class MVTecMetric(BaseMetric):
    def __init__(self, 
                 eval_PRO=True, non_partial_AUC=False, eval_threshold_step=500,
                 collect_device = 'cpu', prefix = None, collect_dir = None) -> None:
        self.eval_PRO = eval_PRO
        self.non_partial_AUC = non_partial_AUC
        self.eval_threshold_step = eval_threshold_step
        super().__init__(collect_device, prefix, collect_dir)

    def process(self, data_batch, data_samples) -> None:
        for i, gt_info in enumerate(data_batch["data_samples"]):
            self.results.append({
                "gt_sem_seg": gt_info.gt_sem_seg.sem_seg.numpy(),
                "gt": gt_info.gt,
                "score_map": data_samples[i]["score_map"],
                "image_score": data_samples[i]["image_score"]
            })

    def compute_metrics(self, results: list) -> dict:
        score_map = np.concatenate([each["score_map"] for each in self.results], axis=0)
        image_score = np.concatenate([each["image_score"] for each in self.results], axis=0)
        # Normalization
        max_score, min_score = score_map.max(), score_map.min()
        print_log(
            msg=f"image_score_max: {image_score.max():.4f} image_score_min: {image_score.min():.4f}, image_score_mean: {image_score.mean():.4f}",
            logger="current")
        print_log(
            msg=f"score_map_max:{max_score:.4f} score_map_min:{min_score:.4f} score_map_mean: {score_map.mean():.4f}",
            logger="current")
        score_map = (score_map - min_score) / (max_score - min_score)
        print_log(
            msg="score_map归一化完毕!",
            logger="current")
        # calculate image-level ROC AUC score
        gt_list = np.asarray([each["gt"] for each in self.results])
        img_auroc = compute_roc_score(gt_list, image_score, self.eval_threshold_step, self.non_partial_AUC)
        # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(gt_list, image_score)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]
        print_log(
            msg=f"F1 image:{f1.max():.4f} threshold:{threshold:.4f}",
            logger="current")
        gt_mask = np.asarray([each["gt_sem_seg"] for each in self.results], dtype=np.int64).squeeze()
        per_pixel_auroc = compute_roc_score(
            gt_mask.flatten(),
            score_map.flatten(), self.eval_threshold_step, self.non_partial_AUC)
        # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(),
                                                        score_map.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]
        print_log(
            msg=f"F1 pixel:{f1.max():.4f} threshold:{threshold:.4f}",
            logger="current")
        total_PRO = compute_pro_score(gt_mask, score_map, self.eval_threshold_step, self.non_partial_AUC) if self.eval_PRO else None
        return {
            "Img_AUROC": img_auroc,
            "Pixel_AUROC": per_pixel_auroc,
            "PRO_score": total_PRO
        }

def compute_roc_score(y_true: np.ndarray,
                      amaps: np.ndarray,
                      steps=500,
                      non_partial_AUC=False) -> float:
    # fprs, tprs = compute_roc(masks, amaps, steps)
    fprs, tprs, thresholds = roc_curve(y_true, amaps)
    return compute_non_partial_auc(fprs, tprs) if non_partial_AUC else auc(fprs,
                                                                           tprs)

def compute_non_partial_auc(fpr, pro, at_fpr=1.0):
    acut = 0.  # area cut
    area = 0.  # area all
    assert 1 < len(pro)
    assert len(fpr) == len(pro)
    for i in range(len(fpr)):
        # calculate bin_size
        if len(fpr) - 1 != i:
            fpr_right = fpr[i + 1]
        else:
            fpr_right = 1.0
        b_left = (fpr[i] - fpr[i - 1]) / 2
        b_right = (fpr_right - fpr[i]) / 2
        if 0 == i:  # left-end
            b = fpr[i] + b_right
        elif len(fpr) - 1 == i:  # right-end
            b = b_left + 1. - fpr[i]
        else:
            b = b_left + b_right
        # calculate area
        if fpr[i] + b_right > at_fpr:
            b_cut = max(0, at_fpr - fpr[i] + b_left)  # bin cut
            acut += b_cut * pro[i]
        else:
            acut += b * pro[i]
        area += b * pro[i]
    return acut / at_fpr

def compute_pro_score(y_true: np.ndarray,
                    amaps: np.ndarray,
                    steps=500,
                    non_partial_AUC=False) -> float:
    fprs, pros = compute_pro(y_true, amaps, steps)
    return compute_non_partial_auc(
        rescale(fprs), rescale(pros)) if non_partial_AUC else auc(
            rescale(fprs),
            rescale(pros))  # compute_non_partial_auc(fprs, rescale(pros), 0.3)

def compute_pro(y_true: np.ndarray, amaps: np.ndarray, steps=500) -> float:
    y_true = y_true.squeeze()

    pros = []
    fprs = []
    for th in tqdm(get_thresholds(amaps, steps, True,
                                  True)):  # thresholds[::-1]:#
        binary_amaps = amaps.squeeze() > th
        """
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                TP_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pro.append(TP_pixels / region.area)
        pros.append(np.mean(pro))"""
        pros.append(compute_pro_(y_true, binary_amaps, 'mean'))

        inverse_masks = 1 - y_true
        FP_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = FP_pixels / inverse_masks.sum()
        fprs.append(fpr)
        if fpr > 0.3: break

    # print(np.array(list(zip(pros,fprs))))
    fprs = np.array(fprs)
    pros = np.array(pros)
    return fprs, pros

def compute_pro_(y_true: np.ndarray, binary_amaps: np.ndarray,
                 method='mean') -> float:
    pros = []
    for binary_amap, mask in zip(binary_amaps, y_true):
        per_region_tpr = []
        for region in measure.regionprops(measure.label(mask)):
            axes0_ids = region.coords[:, 0]
            axes1_ids = region.coords[:, 1]
            TP_pixels = binary_amap[axes0_ids, axes1_ids].sum()
            per_region_tpr.append(TP_pixels / region.area)
        if method == 'mean' and per_region_tpr:
            pros.append(np.mean(per_region_tpr))
        else:
            pros.extend(per_region_tpr)
    return np.mean(pros)

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def get_thresholds(t: np.ndarray, num_samples=1000, reverse=False, opt=True):
    if opt:
        # use the worst-case for efficient determination of thresholds
        max_idx = t.reshape(t.shape[0], -1).max(1).argmax(0)
        t = t[max_idx].flatten()
        # return [kthvalue(t, max(1, math.floor(t.size * i / num_samples)-1)-1)
        #            for i in range(num_samples, 0, -1)]
        r = np.linspace(0, t.size - 1, num=num_samples).astype(int)
        if reverse: r = r[::-1]
        t.sort()
        return t[r]
        # idx = np.argsort(t)
        # return [t[idx[max(1, math.floor(t.size * i / num_samples)-1)-1]] for i in range(num_samples, 0, -1)]
    else:
        # return [kthvalue(t.flatten(), max(1, math.floor(t.size * i / num_samples)))
        #            for i in range(num_samples, 0, -1)]

        r = np.linspace(t.min(), t.max(), num=num_samples)
        if reverse: r = r[::-1]
        return r