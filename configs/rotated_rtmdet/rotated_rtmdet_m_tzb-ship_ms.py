_base_ = "./rotated_rtmdet_m-coco_pretrain-3x-dota_ms.py"

model = dict(
    bbox_head=dict(num_classes=1),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05, # TODO:
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000)) # 2000

# load_from = 'https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_l-coco_pretrain-3x-dota_ms/rotated_rtmdet_l-coco_pretrain-3x-dota_ms-06d248a2.pth'  # noqa

dataset_type = "TzbShipDataset"
data_root = 'data/Tianzhi/ship/'

train_dataloader = dict(
    batch_size=2, 
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file='train_split_Bk/annfiles/',
        data_prefix=dict(img_path='train_split_Bk/images/'),
        filter_cfg=dict(filter_empty_gt=True), # TODO: True
        data_root=data_root))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file='val_split/annfiles/',
        data_prefix=dict(img_path='val_split/images/'),
        data_root=data_root))

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file='val_split/annfiles/',
        data_prefix=dict(img_path='val_split/images/'),
        data_root=data_root))

# learning rate
checkpoint_config = dict(interval=10)
train_cfg = dict(val_interval=2)

optim_wrapper = dict(optimizer=dict(lr=0.001 / 16))
val_evaluator = dict(metric='f1_score', iou_thrs=0.1)
test_evaluator = dict(metric='f1_score', iou_thrs=0.1)
