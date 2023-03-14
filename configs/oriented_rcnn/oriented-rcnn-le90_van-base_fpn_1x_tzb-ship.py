_base_ = './oriented-rcnn-le90_van-base_fpn_1x_dota.py'

angle_version = 'le90'
model = dict(
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    rpn_head=dict(
        bbox_coder=dict(
            angle_version=angle_version,
    )),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
            bbox_coder=dict(
                angle_version=angle_version))))

dataset_type = "TzbShipDataset"
data_root = 'data/Tianzhi/ship/'
train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='train_split_Bk/annfiles/',
        data_prefix=dict(img_path='train_split_Bk/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        data_root=data_root))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='val_split/annfiles/',
        data_prefix=dict(img_path='val_split/images/'),
        data_root=data_root))

train_cfg = dict(val_interval=2)
checkpoint_config = dict(interval=6)