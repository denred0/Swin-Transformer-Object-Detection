_base_ = [
    '../_base_/models/cascade_mask_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                # norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                # norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                # norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]))

dataset_type = 'CocoDataset'
classes = ('0',)
data_root = 'denred0_data/prepare_dataset/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='Resize', img_scale=(736, 736), keep_ratio=True),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 480), (512, 512), (544, 544), (576, 576),
                                 (608, 608), (640, 640), (672, 672), (704, 704),
                                 (736, 736)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(480, 480), (512, 512), (544, 544), (576, 576),
                                 (608, 608), (640, 640), (672, 672), (704, 704),
                                 (736, 736)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(608, 608),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 480), (512, 512), (544, 544), (576, 576),
                                 (608, 608), (640, 640), (672, 672), (704, 704),
                                 (736, 736)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(736, 736),
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/' + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/' + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/' + 'annotations/test.json',
        img_prefix=data_root + 'test/',
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# data = dict(train=dict(pipeline=train_pipeline))


optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)

# load_from = 'denred0_checkpoints/cascade_mask_rcnn_swin_base_patch4_window7.pth'
# 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'

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


# Difference between resume_from and load_from: resume_from loads both the
# model weights and optimizer status, and the epoch is also inherited from
# the specified checkpoint. It is usually used for resuming the training
# process that is interrupted accidentally. load_from only loads the model
# weights and the training epoch starts from 0. It is usually used for
# finetuning.
