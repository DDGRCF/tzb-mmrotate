import numpy as np
import os
import torch
from mmengine.hooks import Hook

from mmrotate.registry import HOOKS
from mmrotate.structures import rbox2qbox


@HOOKS.register_module()
class RecorderOutputHook(Hook):
    def __init__(self, interval=1, save_name="recorder_output"):
        self.interval = interval
        self.save_name = save_name
    
    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not hasattr(self, "save_path"):
            self.save_path = runner.work_dir + "/" + self.save_name
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path) 

        for output in outputs:
            img_id = output.img_id
            scale_factor = output.scale_factor
            pred_instances = output._pred_instances
            bboxes = pred_instances.bboxes
            labels = pred_instances.labels
            scores = pred_instances.scores
            if bboxes.shape[-1] == 5:
                bboxes = rbox2qbox(bboxes)
            scale_factor = bboxes.new_tensor(scale_factor)[None]
            scale_factor = scale_factor.repeat(1, 4)
            bboxes /= scale_factor
            bboxes = bboxes.detach().cpu().numpy().tolist()
            labels = labels.detach().cpu().numpy().tolist()
            scores = scores.detach().cpu().numpy().tolist()
            with open(self.save_path + "/" + img_id + ".txt", "w") as f:
                for bbox, label, score in zip(bboxes, labels, scores):
                    bbox = list(map(str, bbox))
                    label = str(label)
                    score = str(score)
                    bbox.append(label)
                    bbox.append(score)
                    line = ' '.join(bbox) + '\n'
                    f.write(line)

