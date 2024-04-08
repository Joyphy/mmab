from typing import Optional

import os
import os.path as osp
import matplotlib
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.segmentation import mark_boundaries

from mmab.registry import HOOKS
from mmdet.engine.hooks import DetVisualizationHook
from mmengine.utils import mkdir_or_exist
from mmengine.dist import master_only

@HOOKS.register_module()
class ScoreMapVisualizationHook(DetVisualizationHook):
    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 score_thr: float = 0.5,
                 show: bool = False,
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 backend_args: dict = None):
        super().__init__(draw, interval, score_thr, show, wait_time, test_out_dir, backend_args)

    def after_test_iter(self, runner, batch_idx: int, data_batch: dict, outputs) -> None:
        if self.draw is False:
            return

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        for i, gt_info in enumerate(data_batch["data_samples"]):
            self._test_index += 1
            score_map = outputs.score_map[i]
            max_score = score_map.max()
            min_score = score_map.min()
            score_map = (score_map - min_score) / (max_score - min_score)
            plot_fig(
                test_img=data_batch["inputs"][i].numpy(),
                scores=score_map,
                gts=gt_info.gt_sem_seg.sem_seg.numpy(),
                threshold=self.score_thr,
                save_dir=self.test_out_dir,
                save_name=f"{self._test_index:>05d}_" + osp.basename(gt_info.img_path),
            )

def denormalization(x):
    x = x.transpose(1, 2, 0)
    return x

def plot_fig(test_img,
             scores,
             gts,
             threshold,
             save_dir,
             save_name,
             save_pic=True):
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    if gts is not None:
        with_gt = 1
    else:
        with_gt = 0
    img = test_img
    img = denormalization(img)
    heat_map = scores * 255
    mask = scores
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask *= 255
    vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
    fig_img, ax_img = plt.subplots(1, 4 + with_gt, figsize=(12, 3))
    fig_img.subplots_adjust(right=0.9)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for ax_i in ax_img:
        ax_i.axes.xaxis.set_visible(False)
        ax_i.axes.yaxis.set_visible(False)
    ax_img[0].imshow(img)
    ax_img[0].title.set_text('Image')
    if with_gt:
        gt = gts.transpose(1, 2, 0).squeeze()
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
    ax = ax_img[with_gt + 1].imshow(heat_map, cmap='jet', norm=norm)
    ax_img[with_gt + 1].imshow(img, cmap='gray', interpolation='none')
    ax_img[with_gt + 1].imshow(
        heat_map, cmap='jet', alpha=0.5, interpolation='none')
    ax_img[with_gt + 1].title.set_text('Predicted heat map')
    ax_img[with_gt + 2].imshow(mask, cmap='gray')
    ax_img[with_gt + 2].title.set_text('Predicted mask')
    ax_img[with_gt + 3].imshow(vis_img)
    ax_img[with_gt + 3].title.set_text('Segmentation result')
    left = 0.92
    bottom = 0.15
    width = 0.015
    height = 1 - 2 * bottom
    rect = [left, bottom, width, height]
    cbar_ax = fig_img.add_axes(rect)
    cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
    cb.ax.tick_params(labelsize=8)
    font = {
        'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 8,
    }
    cb.set_label('Anomaly Score', fontdict=font)
    if save_pic:
        save_name = os.path.join(save_dir, save_name)
        fig_img.savefig(save_name, dpi=100)
    else:
        plt.show()
    plt.close()
    return