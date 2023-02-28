_base_ = './oriented-rcnn-le90_r50_fpn_1x_dota.py'

pretrained = 'https://download.openmmlab.com/mmclassification/v0/van/van-base_8xb128_in1k_20220501-6a4cc31b.pth'

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
    type='mmdet.FasterRCNN',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375], # TODO:
        bgr_to_rgb=True),
    backbone=dict(
        _delete_=True,
        type='VAN',
        frozen_stages=1,
        arch=arch,
        out_indices=(0, 1, 2, 3),
        drop_rate=0.,
        drop_path_rate=0.2,
        norm_eval=True,
        norm_cfg=dict(type='LN', requires_grad=True),
        block_cfgs=dict(
            norm_cfg=dict(type='BN', requires_grad=True)),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        in_channels=van_zoo[arch]["outs"]),
    rpn_head=dict(
        bbox_coder=dict(
            angle_version=angle_version,
        )),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                angle_version=angle_version)))
    )

train_cfg = dict(val_interval=4)

# backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
# backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
# embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# custom_keys = {
#     'backbone': dict(lr_mult=0.1, decay_mult=1.0),
#     'backbone.patch_embed.norm': backbone_norm_multi,
#     'backbone.norm': backbone_norm_multi,
#     'absolute_pos_embed': backbone_embed_multi,
#     'relative_position_bias_table': backbone_embed_multi,
#     'query_embed': embed_multi,
#     'query_feat': embed_multi,
#     'level_embed': embed_multi
# }
custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
             'relative_position_bias_table': dict(decay_mult=0.),
             'norm': dict(decay_mult=0.)}

# depths = van_zoo[arch]["depths"]
# custom_keys.update({
#     f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
#     for stage_id, num_blocks in enumerate(depths)
#     for block_id in range(num_blocks)
# })
# custom_keys.update({
#     f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
#     for stage_id in range(len(depths) - 1)
# })


# # optimizer
optim_wrapper = dict(
    clip_grad=dict(max_norm=5.0),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    paramwise_cfg=dict(_delete_=True, custom_keys=custom_keys, norm_decay_mult=0.0))
