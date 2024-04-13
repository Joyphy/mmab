# patchcore配置文件
# 常用自定义参数
user_scale = (256, 256)                # 训练尺寸
user_crop_size = (256, 256)            # 随机裁切尺寸
user_score_thr = 0.5                   # 归一化阈值, 仅画图时生效
user_origin_score_thr = None           # 原始阈值, 不为None时会屏蔽归一化阈值, 仅画图时生效
user_random_seed = 3                   # 全局随机种子
user_do_eval = True                    # 是否进行评估
user_eval_PRO = True                   # 是否进行PRO评估, 没有mask缺陷测试集时可关闭
user_train_cfg = dict(                 # 训练相关参数
    feat_concat=[1, 2],                # 合并哪些特征层的索引进行检测，大特征图检测小缺陷，小特征图检测大缺陷
    pool_size=3,                       # 局部特征融合范围, 即pooling大小, 只能为奇数
    gaussian_blur=True,                # 是否将分数图进行高斯模糊, 可以使缺陷mask边缘平滑一些, 指标贡献很低
    sigma=4.0,                         # 值越大, 高斯模糊平均范围越大
    n_neighbors=9,                     # image_score参考多少topk分数, 较大幅度影响image_score值
    k_ratio=10)                        # memory_bank特征保留比例
user_is_resize_mask = True             # 是否将推理出的分数图还原成原始图片大小, 推理参数, 仅影响推理

# 配置文件归属参数
default_scope = "mmab"
runner_type = "mmab.MemoryBankRunner"
randomness = dict(seed=user_random_seed)
train_cfg = dict(type="MemoryBankTrainLoop", do_eval=user_do_eval)
val_cfg = dict(type="ValLoop")
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook', save_param_scheduler=False, save_optimizer=False),
    logger=dict(interval=1, type='LoggerHook'),
    param_scheduler=None,
    visualization=dict(type='ScoreMapVisualizationHook', score_thr=user_score_thr, origin_score_thr=user_origin_score_thr),
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
        num=0,
        data_root="/root/workspace/datasets/bottle/train",
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=user_scale, type='Resize'),
            dict(crop_size=user_crop_size, type="mmdet.RandomCrop"),
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
        data_root="/root/workspace/datasets/bottle/test",
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type="LoadMaskFromFile"),
            dict(keep_ratio=False, scale=user_scale, type='Resize'),
            dict(crop_size=user_crop_size, type="mmdet.RandomCrop"),
            dict(
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction', 'gt'),
                type='mmdet.PackDetInputs'),
        ])
)
val_evaluator = dict(
    type='MVTecMetric',
    eval_PRO=user_eval_PRO,
    non_partial_AUC=False,
    eval_threshold_step=500)

test_dataloader = val_dataloader
test_evaluator = val_evaluator
test_cfg = dict(type="TestLoop")

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
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3)),
    train_cfg=user_train_cfg,
    test_cfg=dict(
        mask_thr=0.0,
        is_origin_mask=True,
        is_resize_mask=user_is_resize_mask
    )
)

test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=user_scale, type='Resize'),
    dict(
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
        type='mmdet.PackDetInputs'
    )
]

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))