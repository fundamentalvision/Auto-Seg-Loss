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

# optimizer_config = dict()
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
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
                     target_metric='BF1', 
                     drop_bg=False,
                     num_class=21, 
                     theta=[0.8113003373146057, 0.8526162505149841, 0.17164576053619385, 0.03164796531200409, 0.20997394621372223, 0.44903290271759033, 0.07847517728805542, 0.8753038048744202], 
                     parameterization='bezier', 
                     loss_weight=1.0)),
            auxiliary_head=dict(
                num_classes=21,
                loss_decode=dict(
                    _delete_=True,
                    type='AutoSegLoss', 
                    target_metric='BF1', 
                    drop_bg=False,
                    num_class=21, 
                    theta=[0.8113003373146057, 0.8526162505149841, 0.17164576053619385, 0.03164796531200409, 0.20997394621372223, 0.44903290271759033, 0.07847517728805542, 0.8753038048744202], 
                    parameterization='bezier', 
                    loss_weight=0.4)))






