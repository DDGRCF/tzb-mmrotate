_base_ = ['./oriented-rcnn-le90_r50_fpn_1x_dota.py']

angle_version = 'le90'

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

train_cfg = dict(val_interval=2)
checkpoint_config = dict(interval=3)


optim_wrapper = dict(
    #type='AmpOptimWrapper',
    optimizer=dict(lr=0.01)
)