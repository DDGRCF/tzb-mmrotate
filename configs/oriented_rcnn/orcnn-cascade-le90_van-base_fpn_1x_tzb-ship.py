_base_ = './orcnn-cascade-le90_r50_fpn_1x_tzb-ship.py'

arch = 'base'
pretrained = 'https://download.openmmlab.com/mmclassification/v0/van/van-base_8xb128_in1k_20220501-6a4cc31b.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='VAN',
        frozen_stages=1,
        arch=arch,
        out_indices=(0, 1, 2, 3),
        drop_rate=0.,
        drop_path_rate=0.2,
        norm_cfg=dict(type='LN', requires_grad=True),
        block_cfgs=dict(
            norm_cfg=dict(type='BN', requires_grad=True)),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        in_channels=[64, 128, 320, 512])
)


custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
             'relative_position_bias_table': dict(decay_mult=0.),
             'norm': dict(decay_mult=0.)}

optim_wrapper = dict(
    clip_grad=dict(max_norm=5.0),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    paramwise_cfg=dict(_delete_=True, custom_keys=custom_keys, norm_decay_mult=0.0))