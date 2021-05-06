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

target_metric = 'mIoU'


mu_lr = 0.20
mu = [0.25, 0.25, 0.3333, 0.3333, 0.5, 0.5, 0.25, 0.25, 0.3333, 0.3333, 0.5, 0.5]
sigma = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
num_pieces = 2
num_samples = 32
sample_times = 12
update_per_sample = 60
clip_eps = 0.2
baseline = 'mu' # Choice: 'mean' or 'mu'
num_models_per_gpu = 4
lr_lambda = None

train_iters = 1000


log_config = dict(interval=-1)


load_from = None
resume_from = None
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
optimizer_config = dict()

# dataset settings
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(512, 128), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(128, 128), cat_max_ratio=0.75),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size=(128, 128), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 128),
                # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='CenterCrop', crop_size=(128, 128)),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
                ])
        ])
)


norm_cfg = dict(type='BN', requires_grad=True)
model = dict(backbone=dict(norm_cfg=norm_cfg),
             decode_head=dict(
                 num_classes=21,
                 norm_cfg=norm_cfg,
                 loss_decode=dict(
                     type='StraightLoss', use_sigmoid=False, loss_weight=1.0)),
            auxiliary_head=dict(
                num_classes=21,
                norm_cfg=norm_cfg,
                loss_decode=dict(type='StraightLoss', use_sigmoid=False, loss_weight=0.4)))













