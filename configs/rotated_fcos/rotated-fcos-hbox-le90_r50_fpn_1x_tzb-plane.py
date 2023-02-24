_base_ = './rotated-fcos-hbox-le90_r50_fpn_1x_dota.py'

model = dict(
    bbox_head=dict(
        num_classes=1))

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