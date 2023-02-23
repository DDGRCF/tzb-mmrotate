# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.registry import DATASETS
from .dota import DOTADataset


@DATASETS.register_module()
class TzbShipDataset(DOTADataset):

    METAINFO = {
        'classes': ('ship', ),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42)]
    }
