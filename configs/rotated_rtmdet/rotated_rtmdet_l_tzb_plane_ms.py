_base_ = "./rotated_rtmdet_l-coco_pretrain-3x-dota_ms.py"

model = dict(
    bbox_head=dict(num_classes=1))

dataset_type = "TzbPlaneDataset"
data_root = 'data/Tianzhi/plane_split/'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        data_root=data_root))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        data_root=data_root))

checkpoint_config = dict(interval=10)
evaluation = None
