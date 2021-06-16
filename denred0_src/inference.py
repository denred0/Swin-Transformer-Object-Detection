import cv2
import numpy as np
import csv

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from utils import get_all_files_in_folder
from pathlib import Path
from tqdm import tqdm


def inference(images_dir, output_folder, config_file, checkpoint_file, images_ext, conf_threshold):
    device = 'cuda:0'
    COLOR = (255, 0, 0)
    classID = 0

    model = init_detector(config_file, checkpoint_file, device=device)

    files = get_all_files_in_folder(images_dir, images_ext)
    submission = []

    for file in tqdm(files):
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)
        result = inference_detector(model, file)

        bboxes = result[0][0]
        boxes_valid = []
        for box in bboxes:
            if box[4] > conf_threshold:
                boxes_valid.append(box)

        box_string = 'no_box'
        if len(boxes_valid) > 0:
            box_string = ''
            for box in boxes_valid:
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), COLOR, 2)

                box_string += str(int(box[0])) + ' ' + str(int(box[1])) + ' ' + str(int(box[2])) + ' ' + str(
                    int(box[3])) + ';'

        if box_string == 'no_box':
            string_result = str(file.stem) + ',' + str(box_string) + ',' + str(classID)
        else:
            string_result = str(file.stem) + ',' + str(box_string[:-1]) + ',' + str(classID)

        submission.append(string_result)

        cv2.imwrite(str(output_folder) + '/' + str(file.stem) + '.png', image)

    #
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
    config_file = 'denred0_model/configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
    checkpoint_file = 'work_dirs/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/epoch_30.pth'
    images_dir = Path('denred0_data/test')
    output_folder = Path('denred0_data/test_result')
    images_ext = ['*.png']

    conf_threshold = 0.3

    inference(images_dir=images_dir,
              output_folder=output_folder,
              config_file=config_file,
              checkpoint_file=checkpoint_file,
              images_ext=images_ext,
              conf_threshold=conf_threshold)
