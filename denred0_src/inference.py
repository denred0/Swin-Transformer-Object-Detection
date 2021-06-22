import cv2
import numpy as np
import csv
import shutil

from mmdet.apis import init_detector, inference_detector
from pathlib import Path
from tqdm import tqdm

from utils import get_all_files_in_folder


def inference(images_dir, output_images_folder, output_txt_folder, config_file, checkpoint_file, images_ext,
              conf_threshold):
    device = 'cuda:0'
    COLOR = (255, 0, 0)
    classID = 0

    # clear folders
    dirpath = output_images_folder
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    dirpath = output_txt_folder
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    model = init_detector(config_file, checkpoint_file, device=device)

    files = get_all_files_in_folder(images_dir, images_ext)
    submission = []

    for file in tqdm(files):
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)

        result = inference_detector(model, file)

        bboxes = result[0]

        # if not isinstance(bboxes, list):
        #     bboxes = [bboxes]

        boxes_valid = []
        for box in bboxes:
            if box[4] > conf_threshold:
                boxes_valid.append(box)

        box_string = 'no_box'
        txt_detection_result = []
        if len(boxes_valid) > 0:
            box_string = ''
            for box in boxes_valid:
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), COLOR, 2)

                box_string += str(int(box[0])) + ' ' + str(int(box[1])) + ' ' + str(int(box[2])) + ' ' + str(
                    int(box[3])) + ';'

                txt_string = str(classID) + ' ' + str(box[4]) + ' ' + str(box[0]) + ' ' + str(box[1]) + ' ' + str(
                    int(box[2] - box[0])) + ' ' + str(int(box[3] - box[1]))
                txt_detection_result.append(txt_string)

        if box_string == 'no_box':
            string_result = str(file.stem) + ',' + str(box_string) + ',' + str(classID)
        else:
            string_result = str(file.stem) + ',' + str(box_string[:-1]) + ',' + str(classID)

        submission.append(string_result)

        with open(Path(output_txt_folder).joinpath(file.stem + '.txt'), 'w') as f:
            for item in txt_detection_result:
                f.write("%s\n" % item)

        cv2.imwrite(str(output_images_folder) + '/' + str(file.stem) + '.png', image)

    with open('denred0_data/submission.csv', 'w') as f:
        f.write('image_name,PredString,domain\n')
        for item in submission:
            f.write("%s\n" % item)


def create_csv():
    results = []
    with open('denred0_data/submission.csv', 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            results.append(row)

    template = []
    with open('denred0_data/submission_template.csv', 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            template.append(row)

    print(f"results {len(results)}")
    print(f"template {len(template)}")

    # find below 0 values
    nol = []
    for res in results:
        boxes = res[1]
        boxes_list = boxes.split(';')
        for box in boxes_list:
            coords = box.split(' ')
            for coord in coords:
                if '-' in coord:
                    nol.append(coord)
                    # print(coord)

    print(np.unique(sorted(nol)))

    total = []
    for temp in template:
        for res in results:
            if temp[0] == res[0]:
                str_total = str(temp[0]) + ',' + str(temp[1]) + ',' + str(res[1])
                total.append(str_total)

    print(f"total {len(total)}")

    with open('denred0_data/sub.csv', 'w') as f:
        for item in total:
            f.write("%s\n" % item)


if __name__ == '__main__':
    config_file = 'denred0_model/configs/swin/wheat/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco_resized_without_mask_head.py'
    checkpoint_file = 'work_dirs/wheat/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco_resized_aug/epoch_6.pth'

    # # eval
    # images_dir = Path('denred0_data/eval/img_source')
    # output_images_folder = Path('denred0_data/eval/img_result')
    # output_txt_folder = Path('denred0_data/eval/txt_result')
    # images_ext = ['*.jpg']

    # inference
    images_dir = Path('denred0_data/test')
    output_images_folder = Path('denred0_data/test_result/img_result')
    output_txt_folder = Path('denred0_data/test_result/txt_result')
    images_ext = ['*.png']

    conf_threshold = 0.3

    inference(images_dir=images_dir,
              output_images_folder=output_images_folder,
              output_txt_folder=output_txt_folder,
              config_file=config_file,
              checkpoint_file=checkpoint_file,
              images_ext=images_ext,
              conf_threshold=conf_threshold)

    create_csv()
