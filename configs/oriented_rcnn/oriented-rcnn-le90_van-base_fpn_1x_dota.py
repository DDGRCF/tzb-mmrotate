_base_ = './oriented-rcnn-le90_r50_fpn_1x_dota.py'

pretrained = 'https://download.openmmlab.com/mmclassification/v0/van/van-base_8xb128_in1k_20220501-6a4cc31b.pth'

test_pipeline = _base_.test_pipeline
dataset_type = _base_.dataset_type
data_root = _base_.data_root

van_zoo = dict(
    tiny={
        "outs": [32, 64, 160, 256],
        "depths": [3, 3, 5, 2]},
    small={
        "outs": [64, 128, 320, 512],
        "depths": [2, 2, 4, 2]},
    base={
        "outs": [64, 128, 320, 512],
        "depths": [3, 3, 12, 3]},
    large={
        "outs": [64, 128, 320, 512],
        "depths": [3, 5, 27, 3]})

arch = 'base'

angle_version = 'le90'
model = dict(
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    backbone=dict(
        _delete_=True,
        type='VAN',
        frozen_stages=1,
        arch=arch,
        out_indices=(0, 1, 2, 3),
        drop_rate=0.,
        drop_path_rate=0.2,
        # norm_cfg=dict(type='LN', requires_grad=True),
        # block_cfgs=dict(
        #     norm_cfg=dict(type='BN', requires_grad=True)),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        in_channels=van_zoo[arch]["outs"]),
    rpn_head=dict(
        bbox_coder=dict(
            angle_version=angle_version)),
    roi_head=dict(
       bbox_coder=dict(
           angle_version=angle_version)))


train_cfg = dict(val_interval=6)
checkpoint_config = dict(interval=3)

custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
             'relative_position_bias_table': dict(decay_mult=0.),
             'norm': dict(decay_mult=0.)}


test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images/'),
        img_shape=(1024, 1024),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/dota/Task1')

base_lr = 0.0001
# optimizer
optim_wrapper = dict(
    clip_grad=dict(max_norm=5.0),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    paramwise_cfg=dict(_delete_=True, custom_keys=custom_keys, norm_decay_mult=0.0))

max_epochs = 12
