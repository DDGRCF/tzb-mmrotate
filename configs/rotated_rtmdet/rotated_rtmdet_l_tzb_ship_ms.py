# _base_ = "./rotated_rtmdet_l-coco_pretrain-3x-dota_ms.py"
_base_ = "./rotated_rtmdet_l-3x-dota_ms.py"

model = dict(
    data_preprocessor=dict(
        mean=[103.53, 116.28, 123.675],
        std=[1., 1., 1.], bgr_to_rgb=False),
    bbox_head=dict(num_classes=1))

dataset_type = "TzbShipDataset"
data_root = 'data/Tianzhi/ship/'

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

checkpoint_config = dict(interval=10)
evaluation = None