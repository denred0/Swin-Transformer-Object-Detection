_base_ = [
    '../_base_/models/mask_rcnn_swin_fpn_without_mask_head.py',
    '../_base_/datasets/coco_instance_resized.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(in_channels=[96, 192, 384, 768]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#
#
# mean_255: tensor([54.3459, 82.1696, 82.7305])
# std_255:  tensor([51.1432, 62.0632, 64.5409])

# img_norm_cfg = dict(
#     mean=[54.3459, 82.1696, 82.7305], std=[51.1432, 62.0632, 64.5409], to_rgb=True)


# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 480), (512, 512), (544, 544), (576, 576),
                                 (608, 608), (640, 640), (672, 672), (704, 704),
                                 (736, 736), (768, 768), (800, 800), (832, 832), (864, 864)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 400), (500, 500), (600, 600)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(608, 608),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 480), (512, 512), (544, 544),
                                 (576, 576), (608, 608), (640, 640),
                                 (672, 672), (704, 704), (736, 736),
                                 (768, 768), (800, 800), (832, 832), (864, 864)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)

evaluation = dict(metric=['bbox'])

load_from = 'denred0_checkpoints/mask_rcnn_swin_small_patch4_window7.pth'
# resume_from = 'work_dirs/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco_resized/epoch_5.pth'

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
