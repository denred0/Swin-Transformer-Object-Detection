from yolo_to_coco import *
from train_test_split import *

if __name__ == '__main__':

    # split yolo_dataset on train, val, test
    val_part = 0.2
    test_part = 0.5
    img_ext = 'jpg'
    data_dir = Path('denred0_data/prepare_dataset')
    create_train_val_test(data_dir=data_dir,
                          val_part=val_part,
                          test_part=test_part,
                          img_ext=img_ext)

    # create train.txt, val.txt, test.txt for coco dataset format
    folders = ['train', 'val', 'test']
    for folder in folders:
        dir_path = Path(data_dir).joinpath(folder).joinpath('annotations')
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        image_folder = '/home/vid/hdd/projects/PycharmProjects/Swin-Transformer-Object-Detection/denred0_data/prepare_dataset/' + folder
        path_txt = str(Path(data_dir).joinpath(folder).joinpath('annotations').joinpath(folder + '.txt'))
        replacer(image_folder, path_txt)

    # create annotation json coco format
    for folder in folders:
        opt = get_args()
        opt.path = '/home/vid/hdd/projects/PycharmProjects/Swin-Transformer-Object-Detection/denred0_data/prepare_dataset/' + \
                   folder + '/annotations/' + folder + '.txt'
        opt.debug = False
        opt.output = folder

        output_name = opt.output
        output_path = 'denred0_data/prepare_dataset/' + folder + '/annotations/' + output_name + '.json'

        print("Start!")

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
