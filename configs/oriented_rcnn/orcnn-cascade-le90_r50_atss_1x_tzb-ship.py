_base_ = './oriented-rcnn-le90_r50_fpn_1x_tzb-ship.py'

angle_version = _base_.angle_version

model = dict(
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    type='RotatedATSSAssigner',
                    topk=9,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                sampler=dict(
                    type='mmdet.RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='RotatedATSSAssigner',
                    topk=9,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                sampler=dict(
                    type='mmdet.RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='RotatedATSSAssigner',
                    topk=9,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                sampler=dict(
                    type='mmdet.RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ])
)
