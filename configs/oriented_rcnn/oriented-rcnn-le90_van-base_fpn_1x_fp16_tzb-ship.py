_base_ = './oriented-rcnn-le90_van-base_fpn_1x_tzb-ship.py'

optim_wrapper = dict(type='AmpOptimWrapper', optimizer=dict(lr=0.00005))

train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=4, num_workers=4)