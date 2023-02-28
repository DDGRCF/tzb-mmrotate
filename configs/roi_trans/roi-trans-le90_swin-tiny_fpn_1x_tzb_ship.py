_base_ = './roi-trans-le90_swin-tiny_fpn_1x_dota.py'

dataset_type = "TzbShipDataset"
data_root = 'data/Tianzhi/ship/'

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=[9, 11]),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_split/annfiles/',
        data_prefix=dict(img_path='train_split/images/'),
        filter_cfg=dict(filter_empty_gt=True), # TODO: True
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val_split/annfiles/',
        data_prefix=dict(img_path='val_split/images/'),
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
checkpoint_config = dict(interval=10)

load_from = 'https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_r50_fpn_1x_dota_ms_rr_le90/roi_trans_r50_fpn_1x_dota_ms_rr_le90-fa99496f.pth'

train_cfg = dict(val_interval=4)