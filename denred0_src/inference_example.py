from mmdet.apis import init_detector, inference_detector, show_result_pyplot

config_file = 'denred0_model/configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'work_dirs/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/epoch_30.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
img = 'denred0_data/dataset_0/dataset/0a3a42c94d470fe1952d600af7a3629fccee767925a6512d197d50be3d2deeea.jpg'
result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=0.3)
