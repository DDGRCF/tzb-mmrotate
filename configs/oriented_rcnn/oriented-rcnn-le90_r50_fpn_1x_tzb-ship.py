_base_ = ['./oriented-rcnn-le90_r50_fpn_1x_dota.py']

angle_version = 'le90'
file_client_args = dict(backend='disk')


model = dict(
    data_preprocessor=dict(
        bgr_to_rgb=False),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1)))

dataset_type = 'TzbShipDataset'
data_root = 'data/Tianzhi/ship/'
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type=dataset_type,
        ann_file='train_split/annfiles/',
        data_prefix=dict(img_path='train_split/images/'),
        filter_cfg=dict(filter_empty_gt=False),
        data_root=data_root))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='val_split/annfiles/',
        data_prefix=dict(img_path='val_split/images/'),
        data_root=data_root))

val_evaluator = dict(metric='f1_score', iou_thrs=0.1)

test_pipeline = [
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

test_dataloader = dict(
    _delete_ = True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val_split/annfiles/',
        data_prefix=dict(img_path='val_split/images/'),
        img_shape=(1024, 1024),
        test_mode=True,
        pipeline=test_pipeline))

test_evaluator = val_evaluator

train_cfg = dict(val_interval=2)
checkpoint_config = dict(interval=3)


optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(lr=0.01)
)