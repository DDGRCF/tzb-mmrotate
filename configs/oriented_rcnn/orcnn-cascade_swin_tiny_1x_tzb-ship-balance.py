_base_ = './orcnn-cascade-le90_swin-tiny_fpn_1x_tzb-ship.py'


model = dict(
    test_cfg=dict(
        rcnn=dict(score_thr=0.05)
    )
)

data_root = 'data/Tianzhi/ship/'
train_dataloader = dict(
    dataset=dict(
        ann_file='train_split_Bk/annfiles/',
        data_prefix=dict(img_path='train_split_Bk/images/'),
        filter_cfg=dict(filter_empty_gt=False),
        data_root=data_root))

