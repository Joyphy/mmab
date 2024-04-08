# patchcore配置文件

default_scope = "mmab"
runner_type = "mmab.MemoryBankRunner"
randomness = dict(seed=3)
train_cfg = dict(type="MemoryBankTrainLoop", do_eval=True)
val_cfg = dict(type="ValLoop")
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook', save_param_scheduler=False, save_optimizer=False),
    logger=dict(interval=1, type='LoggerHook'),
    param_scheduler=None,
    visualization=dict(type='ScoreMapVisualizationHook'),
)
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(shuffle=True, type='DefaultSampler'),
    dataset=dict(
        type="MVTecDataset",
        is_train=True,
        is_random=True,
        num=20,
        data_root="/root/workspace/datasets/bottle",
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(256, 256), type='Resize'),
            dict(crop_size=(256, 256), type="mmdet.RandomCrop"),
            dict(type='mmdet.PackDetInputs'),
        ])
)
val_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=dict(
        type="MVTecDataset",
        is_train=False,
        is_random=True,
        num=0,
        data_root="/root/workspace/datasets/bottle",
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type="LoadMaskFromFile"),
            dict(keep_ratio=False, scale=(256, 256), type='Resize'),
            dict(crop_size=(256, 256), type="mmdet.RandomCrop"),
            dict(
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction', 'gt'),
                type='mmdet.PackDetInputs'),
        ])
)
val_evaluator = dict(type='MVTecMetric')

test_dataloader = val_dataloader
test_evaluator = val_evaluator
test_cfg = dict(type="TestLoop")

tnr_checkpoint = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
model = dict(
    type="PatchCore",
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        pad_size_divisor=1,
        std=[58.395, 57.12, 57.375],
        type='mmdet.DetDataPreprocessor'),
    backbone=dict(
        _scope_='mmdet',
        type='ResNet',
        style='pytorch',
        depth=18,
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        # init_cfg=dict(type='Pretrained', checkpoint='https://download.pytorch.org/models/resnet50-11ad3fa6.pth'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3)),
    test_cfg=dict(
        mask_thr=0.2
    )
)

test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(256, 256), type='Resize'),
    dict(
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
        type='mmdet.PackDetInputs'
    )
]

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))