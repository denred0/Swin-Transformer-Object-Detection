import shutil
import os

from sklearn.model_selection import train_test_split
from pathlib import Path


def create_train_val_test(data_dir, val_part=0.2, test_part=0.5, img_ext='jpg'):
    # clear folder

    dataset_dir = Path(data_dir).joinpath('dataset')
    # dataset_folder_name = 'dataset'

    dirpath = Path(data_dir).joinpath('train')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    # clear folder
    dirpath = Path(data_dir).joinpath('val')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    # clear folder
    dirpath = Path(data_dir).joinpath('test')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    train_val_files = list(dataset_dir.rglob('*.' + img_ext))

    train_files, val_test_files = train_test_split(train_val_files, test_size=val_part)

    for name in train_files:
        shutil.copy(name, os.path.join(data_dir, 'train'))
        file_name, file_extension = os.path.splitext(name)
        name_txt = file_name + '.txt'
        shutil.copy(name_txt, os.path.join(data_dir, 'train'))

    if test_part != 0:
        val_files, test_files = train_test_split(val_test_files, test_size=test_part)

        for name in val_files:
            shutil.copy(name, os.path.join(data_dir, 'val'))
            file_name, file_extension = os.path.splitext(name)
            name_txt = file_name + '.txt'
            shutil.copy(name_txt, os.path.join(data_dir, 'val'))

        for name in test_files:
            shutil.copy(name, os.path.join(data_dir, 'test'))
            file_name, file_extension = os.path.splitext(name)
            name_txt = file_name + '.txt'
            shutil.copy(name_txt, os.path.join(data_dir, 'test'))
    else:
        for name in val_test_files:
            shutil.copy(name, os.path.join(data_dir, 'val'))
            file_name, file_extension = os.path.splitext(name)
            name_txt = file_name + '.txt'
            shutil.copy(name_txt, os.path.join(data_dir, 'val'))

        for name in val_test_files:
            shutil.copy(name, os.path.join(data_dir, 'test'))
            file_name, file_extension = os.path.splitext(name)
            name_txt = file_name + '.txt'
            shutil.copy(name_txt, os.path.join(data_dir, 'test'))


if __name__ == '__main__':
    val_part = 0.2
    test_part = 0.5

    img_ext = 'jpg'

    data_dir = Path('denred0_data/train_test_split')

    create_train_val_test(data_dir=data_dir,
                          val_part=val_part,
                          test_part=test_part,
                        img_ext=img_ext)
