from mmdet.apis import init_detector, inference_detector, show_result_pyplot

config_file = 'denred0_model/configs/swin/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco_resized_without_mask_head.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'work_dirs/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco_resized_without_mask_head_old/epoch_12.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
img = 'denred0_data/test/0a9efef7204d68d9aede0f648e14fea58de82bcfe3718733ac0c3071a41b18cd.png'
result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=0.3)
