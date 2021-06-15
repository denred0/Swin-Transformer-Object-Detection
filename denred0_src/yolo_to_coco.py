import os
import glob
import argparse
import cv2
import json
import numpy as np
from tqdm import tqdm

#### create train.txt #########################
parser = argparse.ArgumentParser(
    description="This Python file makes preliminary preparations for the conversion of the outputs obtained in the yolo format to json format.")
parser.add_argument('-i', '--path_image_folder', help="Enter the file path where the images are located.", type=str)
parser.add_argument('-p', '--path_txt', help="Enter the file path of the 'txt' file you want to create.", type=str)
args = parser.parse_args()


def replacer(image_folder, path_txt):
    path = os.path.abspath(image_folder)
    f = open(path_txt, "w")

    for i in glob.glob(path + "/*.jpg"):
        f.write(i + "\n")
    print('----File ".txt" created successfully----')
    f.close()


###############################################

#### create_annotations #######################

def create_image_annotation(file_name, width, height, image_id):
    file_name = file_name.split('/')[-1]  # ~~~.jpg가 file_name이 되도록 문자열 split
    images = {
        'file_name': file_name,
        'height': height,
        'width': width,
        'id': image_id
    }
    return images


def create_annotation_yolo_format(min_x, min_y, width, height, image_id, category_id, annotation_id):
    bbox = (min_x, min_y, width, height)
    area = width * height
    segm = [min_x, min_y, (min_x + width), min_y, (min_x + width), (min_y + height), min_x, (min_y + height), min_x,
            min_y]

    annotation = {
        'id': annotation_id,
        'image_id': image_id,
        'bbox': bbox,
        'area': area,
        'iscrowd': 0,
        'category_id': category_id,
        'segmentation': [segm]
    }

    return annotation


# Create the annotations of the ECP dataset (Coco format)
coco_format = {
    "images": [
        {
        }
    ],
    "categories": [

    ],
    "annotations": [
        {
        }
    ]
}

###############################################

#################################################
# Change the classes depend on your own dataset.#
# Don't change the list name 'Classes'          #
#################################################

# Class에 맞게 바꿔줘야함
classes = [
    "0",
]


# Get 'images' and 'annotations' info
def images_annotations_info(opt):
    path = opt.path
    # path : train.txt or test.txt
    annotations = []
    images = []

    file = open(path, "r")
    read_lines = file.readlines()
    file.close()

    image_id = 0
    annotation_id = 1  # In COCO dataset format, you must start annotation id with '1'

    for line in tqdm(read_lines):
        # Check how many items have progressed
        # if image_id % 1000 == 0:
        #     print("Processing " + str(image_id) + " ...")

        line = line.replace('\n', '')
        img_file = cv2.imread(line)

        # read a label file
        label_path = line[:-3] + "txt"
        label_file = open(label_path, "r")
        label_read_line = label_file.readlines()
        label_file.close()

        h, w, _ = img_file.shape

        # Create image annotation
        image = create_image_annotation(line, w, h, image_id)
        images.append(image)

        # yolo format - (class_id, x_center, y_center, width, height)
        # coco format - (annotation_id, x_upper_left, y_upper_left, width, height)
        for line1 in label_read_line:
            label_line = line1
            category_id = int(label_line.split()[0]) + 1  # you start with annotation id with '1'
            x_center = float(label_line.split()[1])
            y_center = float(label_line.split()[2])
            width = float(label_line.split()[3])
            height = float(label_line.split()[4])

            int_x_center = img_file.shape[1] * x_center
            int_y_center = img_file.shape[0] * y_center
            int_width = int(img_file.shape[1] * width)
            int_height = int(img_file.shape[0] * height)

            min_x = int(int_x_center - int_width / 2)
            min_y = int(int_y_center - int_height / 2)
            width = int_width
            height = int_height

            annotation = create_annotation_yolo_format(min_x, min_y, width, height, image_id, category_id,
                                                       annotation_id)
            annotations.append(annotation)
            annotation_id += 1

        image_id += 1  # if you finished annotation work, updates the image id.

    return images, annotations


# Visualize bounding box
def debug(opt):
    path = opt.path
    color_list = np.random.randint(low=0, high=256, size=(len(classes), 3)).tolist()

    # read the file
    file = open(path, "r")
    read_lines = file.readlines()
    file.close()

    for line in read_lines:
        print("Image Path : ", line)
        # read image file
        img_file = cv2.imread(line[:-1])

        # read .txt file
        label_path = line[:-4] + "txt"
        label_file = open(label_path, "r")
        label_read_line = label_file.readlines()
        label_file.close()

        for line1 in label_read_line:
            label_line = line1

            category_id = label_line.split()[0]
            x_center = float(label_line.split()[1])
            y_center = float(label_line.split()[2])
            width = float(label_line.split()[3])
            height = float(label_line.split()[4])

            int_x_center = int(img_file.shape[1] * x_center)
            int_y_center = int(img_file.shape[0] * y_center)
            int_width = int(img_file.shape[1] * width)
            int_height = int(img_file.shape[0] * height)

            min_x = int_x_center - int_width / 2
            min_y = int_y_center - int_height / 2
            width = int(img_file.shape[1] * width)
            height = int(img_file.shape[0] * height)

            print("class name :", classes[int(category_id)])
            print("x_upper_left : ", min_x, '\t', "y_upper_left : ", min_y)
            print("width : ", width, '\t', '\t', "height : ", height)
            print()

            # Draw bounding box
            cv2.rectangle(img_file, (int(int_x_center - int_width / 2), int(int_y_center - int_height / 2)),
                          (int(int_x_center + int_width / 2), int(int_y_center + int_height / 2)),
                          color_list[int(category_id)], 3)

        cv2.imshow(line, img_file)
        delay = cv2.waitKeyEx()

        # If you press ESC, exit
        if delay == 27 or delay == 113:
            break

        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser('Yolo format annotations to COCO dataset format')
    parser.add_argument('-p', '--path', type=str, help='Absolute path for \'train.txt\' or \'test.txt\'')
    parser.add_argument('--debug', action='store_true', help='Visualize bounding box and print annotation information')
    parser.add_argument('--output', type=str, help='Name the output json file')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # args.path_image_folder = '/home/vid/hdd/projects/PycharmProjects/Swin-Transformer-Object-Detection/denred0_data/yolo_to_coco/train/'
    # args.path_txt = 'denred0_data/yolo_to_coco/train.txt'
    # replacer(args.path_image_folder, args.path_txt)

    opt = get_args()
    opt.path = '/home/vid/hdd/projects/PycharmProjects/Swin-Transformer-Object-Detection/denred0_data/yolo_to_coco/train.txt'
    opt.debug = False
    opt.output = 'train'

    output_name = opt.output
    output_path = 'denred0_data/yolo_to_coco/output/' + output_name + '.json'

    print("Start!")

    if opt.debug is True:
        debug(opt)
        print("Debug Finished!")
    else:
        # start converting format
        coco_format['images'], coco_format['annotations'] = images_annotations_info(opt)

        for index, label in enumerate(classes):
            ann = {
                "supercategory": "wheat",
                "id": index + 1,  # Index starts with '1' .
                "name": label
            }
            coco_format['categories'].append(ann)

        with open(output_path, 'w') as outfile:
            json.dump(coco_format, outfile)

        print("Finished!")
