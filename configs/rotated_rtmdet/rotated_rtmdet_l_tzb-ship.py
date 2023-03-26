_base_ = "./rotated_rtmdet_l-coco_pretrain-3x-dota_ms.py"

file_client_args = dict(backend='disk') # TODO:
img_size = (1024, 1024)
pad_val = (0, 0, 0)

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=img_size, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=None),
    dict(
        type='mmdet.Pad', size=img_size,
        pad_val=dict(img=pad_val)),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=img_size, keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad', size=img_size,
        pad_val=dict(img=pad_val)),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=img_size, keep_ratio=True),
    dict(
        type='mmdet.Pad', size=img_size,
        pad_val=dict(img=pad_val)),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
model = dict(
    bbox_head=dict(num_classes=1),
    test_cfg=dict(
        nms_pre=1000, # TODO: ori 2000
        min_bbox_size=0,
        score_thr=0.05, # TODO:
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=1000)) # 2000

dataset_type = "TzbShipDataset"
data_root = 'data/Tianzhi/ship/'
img_suffix = ".png"

train_dataloader = dict(
    batch_size=4, 
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        img_suffix=img_suffix,
        ann_file='train_split/annfiles/',
        data_prefix=dict(img_path='train_split/images/'),
        filter_cfg=dict(filter_empty_gt=False), # TODO: True
        data_root=data_root,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        img_suffix=img_suffix,
        ann_file='val_split/annfiles/',
        data_prefix=dict(img_path='val_split/images/'),
        data_root=data_root,
        pipeline=val_pipeline))

# learning rate
checkpoint_config = dict(interval=1)
checkpoint_config = dict(interval=1)
_base_.default_hooks["checkpoint"]["interval"] = checkpoint_config["interval"]
train_cfg = dict(val_interval=36)

optim_wrapper = dict(type='AmpOptimWrapper', optimizer=dict(lr=0.001 / 8)) # TODO:
val_evaluator = dict(metric='f1_score', iou_thrs=0.1, merge_patches=True, format_only=True, outfile_prefix="test_results")
test_evaluator = dict(metric='f1_score', iou_thrs=0.1, merge_patches=True, format_only=True, outfile_prefix="test_results")
