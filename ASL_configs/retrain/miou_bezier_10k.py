_base_ = [
    '../../configs/_base_/models/deeplabv3plus_r50-d8.py',
    '../../configs/_base_/datasets/pascal_voc12_aug.py', '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/schedules/schedule_20k.py'
]

dist_params = dict(backend='nccl', port=18869)

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
                     target_metric='mIoU', 
                     num_class=21, 
                     theta=[5.69728440e-04, 6.43720450e-01, 3.38589169e-01, 7.05543360e-01, 6.25281252e-01, 7.79551927e-01, 6.90375570e-01, 1.30628900e-02, 8.34170515e-01, 9.62431815e-01, 9.42998269e-01, 9.80038125e-01], 
                     parameterization='bezier', 
                     loss_weight=1.0)),
            auxiliary_head=dict(
                num_classes=21,
                loss_decode=dict(
                     _delete_=True,
                     type='AutoSegLoss', 
                     target_metric='mIoU', 
                     num_class=21, 
                     theta=[5.69728440e-04, 6.43720450e-01, 3.38589169e-01, 7.05543360e-01, 6.25281252e-01, 7.79551927e-01, 6.90375570e-01, 1.30628900e-02, 8.34170515e-01, 9.62431815e-01, 9.42998269e-01, 9.80038125e-01], 
                     parameterization='bezier', 
                     loss_weight=0.4)))
