# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.registry import DATASETS
from .dota import DOTADataset


@DATASETS.register_module()
class TzbPlaneDataset(DOTADataset):

    METAINFO = {
        'classes': ('plane', ),
        'palette': [(165, 42, 42)]
    }

