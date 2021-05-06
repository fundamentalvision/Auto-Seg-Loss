_base_ = [
    '../../configs/_base_/models/deeplabv3plus_r50-d8.py',
    '../../configs/_base_/datasets/pascal_voc12_aug.py', '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/schedules/schedule_20k.py'
]

optimizer = dict(
    type='SGD', lr=0.02,
    paramwise_cfg = dict(
        custom_keys={
            'head': dict(lr_mult=10.)}))

optimizer_config = dict()
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
load_from = None
resume_from = None

runner = dict(max_iters=10000)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4
)

model = dict(pretrained='open-mmlab://resnet101_v1c', 
             backbone=dict(depth=101),
             decode_head=dict(
                 num_classes=21,
                 loss_decode=dict(
                     _delete_=True,
                     type='AutoSegLoss', 
                     target_metric='BIoU', 
                     drop_bg=True,
                     tol=5,
                     num_class=21, 
                     theta=[3.65390355e-04, 9.84171689e-01, 9.69491959e-01, 9.92340326e-01, 9.88328516e-01, 9.95552719e-01, 3.54609758e-01, 2.67221928e-01, 5.60551465e-01, 4.35290039e-01, 5.65432191e-01, 6.01448298e-01], 
                     parameterization='bezier', 
                     loss_weight=1.0)),
            auxiliary_head=dict(
                num_classes=21,
                loss_decode=dict(
                    _delete_=True,
                    type='AutoSegLoss', 
                    target_metric='BIoU', 
                    drop_bg=True,
                    tol=5, 
                    num_class=21, 
                    theta=[3.65390355e-04, 9.84171689e-01, 9.69491959e-01, 9.92340326e-01, 9.88328516e-01, 9.95552719e-01, 3.54609758e-01, 2.67221928e-01, 5.60551465e-01, 4.35290039e-01, 5.65432191e-01, 6.01448298e-01], 
                    parameterization='bezier', 
                    loss_weight=0.4)))






