_base_ = './s2anet-le135_r50_fpn_1x_dota.py'


dataset_type = "TzbShipDataset"
data_root = 'data/Tianzhi/ship/'

angle_version = 'le90'
model = dict(
    data_preprocessor=dict(
        mean=[103.53, 116.28, 123.675],
        std=[1., 1., 1.], bgr_to_rgb=False),
    bbox_head_init=dict(num_classes=1),
    bbox_head_refine=[
        dict(
            type='S2ARefineHead',
            num_classes=1,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            frm_cfg=dict(
                type='AlignConv',
                feat_channels=256,
                kernel_size=3,
                strides=[8, 16, 32, 64, 128]),
            anchor_generator=dict(
                type='PseudoRotatedAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=0.11, loss_weight=1.0))
    ])

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='train_split/annfiles/',
        data_prefix=dict(img_path='train_split/images/'),
        data_root=data_root))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='val_split/annfiles/',
        data_prefix=dict(img_path='val_split/images/'),
        data_root=data_root))