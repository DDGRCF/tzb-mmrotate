_base_ = './rotated_rtmdet_l-coco_pretrain-3x-dota_ms.py'

coco_ckpt = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'  # noqa

model = dict(
    backbone=dict(
        deepen_factor=0.67,
        widen_factor=0.75, 
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=coco_ckpt)),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2,
              init_cfg=dict(type='Pretrained', prefix='neck.',
                      checkpoint=coco_ckpt)),
    bbox_head=dict(
        in_channels=192,
        feat_channels=192,
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        init_cfg=dict(
            type='Pretrained', prefix='bbox_head.', checkpoint=coco_ckpt)))

# batch_size = (1 GPUs) x (8 samples per GPU) = 8
train_dataloader = dict(batch_size=4, num_workers=4)
