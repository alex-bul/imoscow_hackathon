import csv
import os
import shutil
import argparse
import zipfile
from preprocessing import DataPreprocessor


def move_images_and_labels(src_image_folder, src_label_folder, dest_image_folder, dest_label_folder):
    # Перемещение картинок
    for filename in os.listdir(src_image_folder):
        src_image_path = os.path.join(src_image_folder, filename)
        dest_image_path = os.path.join(dest_image_folder, filename)

        shutil.move(src_image_path, dest_image_path)

    # Перемещение файлов меток
    for filename in os.listdir(src_label_folder):
        src_label_path = os.path.join(src_label_folder, filename)
        dest_label_path = os.path.join(dest_label_folder, filename)

        shutil.move(src_label_path, dest_label_path)


def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' has been successfully deleted.")
    except Exception as e:
        print(f"Error deleting folder '{folder_path}': {e}")


def delete_files(file_paths):
    try:
        for file_path in file_paths:
            os.remove(file_path)
            print(f"File '{file_path}' has been successfully deleted.")
    except Exception as e:
        print(f"Error deleting files: {e}")


def extract_zip():
    current_dir = os.getcwd()
    yolov8_folder = 'AIWDB_yolov8_sc'  # Specify the folder name
    data_yaml_file = 'data.yaml'  # Specify the file name
    # yolov8
    for file in os.listdir(current_dir):
        if file.endswith(".zip") and "yolov8" in file.lower():
            zip_file_path = os.path.join(current_dir, file)

            # Распаковка архива
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(current_dir)

            # Удаление zip-архива после распаковки
            os.remove(zip_file_path)

            # Перемещение data.yaml
            src_data_yaml = os.path.join(current_dir, data_yaml_file)
            dest_data_yaml = os.path.join(current_dir, yolov8_folder, data_yaml_file)
            shutil.move(src_data_yaml, dest_data_yaml)

    # tensorflow
    for file in os.listdir(current_dir):
        if file.endswith(".zip") and "tensorflow" in file.lower():
            zip_file_path = os.path.join(current_dir, file)

            # Распаковка архива
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(current_dir)

            # Удаление zip-архива после распаковки
            os.remove(zip_file_path)


# функция для переноса картинок и аннотаций в соответствующие выборке папки
def copy_and_remove(csv_file, source_folder, dest_image_folder, dest_label_folder):
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row['filename']
            image_path = os.path.join(source_folder, 'train/images', filename)
            label_path = os.path.join(source_folder, 'train/labels', filename[:-4] + '.txt')

            dest_image_path = os.path.join(dest_image_folder, filename)
            dest_label_path = os.path.join(dest_label_folder, filename[:-4] + '.txt')

            # Проверка, был ли файл уже скопирован
            if not os.path.exists(dest_image_path):
                # Копирование изображения
                shutil.copy(image_path, dest_image_path)
                os.remove(image_path)  # Удаление из исходной папки

            if not os.path.exists(dest_label_path):
                # Копирование файла меток
                shutil.copy(label_path, dest_label_path)
                os.remove(label_path)  # Удаление из исходной папки


def archive_folder(folder_path, output_path):
    try:
        shutil.make_archive(output_path, 'zip', folder_path)
        print(f"Folder '{folder_path}' has been successfully archived to '{output_path}.zip'.")
    except Exception as e:
        print(f"Error archiving folder: {e}")


if __name__ == "__main__":
    extract_zip()

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotations', default='./train/_annotations.csv', type=str)
    parser.add_argument('-o', '--outdir', default='.', type=str)

    args = parser.parse_args()

    preprocessor = DataPreprocessor()

    preprocessor.split_roboflow_dataset(
        annotations_csv=args.annotations,
        output_dir=args.outdir
    )

    # test
    csv_file_test = 'test_split.csv'
    source_folder_test = '.'
    dest_image_folder_test = 'AIWDB_yolov8_sc/test/images'
    dest_label_folder_test = 'AIWDB_yolov8_sc/test/labels'

    # train
    csv_file_train = 'train_split.csv'
    source_folder_train = '.'
    dest_image_folder_train = 'AIWDB_yolov8_sc/train/images'
    dest_label_folder_train = 'AIWDB_yolov8_sc/train/labels'

    # valid
    csv_file_valid = 'val_split.csv'
    source_folder_valid = '.'
    dest_image_folder_valid = 'AIWDB_yolov8_sc/valid/images'
    dest_label_folder_valid = 'AIWDB_yolov8_sc/valid/labels'

    # Проверка и создание папок назначения, если они не существуют
    os.makedirs(dest_image_folder_test, exist_ok=True)
    os.makedirs(dest_label_folder_test, exist_ok=True)

    os.makedirs(dest_image_folder_train, exist_ok=True)
    os.makedirs(dest_label_folder_train, exist_ok=True)

    os.makedirs(dest_image_folder_valid, exist_ok=True)
    os.makedirs(dest_label_folder_valid, exist_ok=True)

    # Вызов функции для копирования файлов и удаления из исходной папки
    copy_and_remove(csv_file_test, source_folder_test, dest_image_folder_test, dest_label_folder_test)
    copy_and_remove(csv_file_train, source_folder_train, dest_image_folder_train, dest_label_folder_train)
    copy_and_remove(csv_file_valid, source_folder_valid, dest_image_folder_valid, dest_label_folder_valid)

    # Указываем пути к исходным и целевым папкам
    src_image_folder = './train/images'
    src_label_folder = './train/labels'
    dest_image_folder = './AIWDB_yolov8_sc/train/images'
    dest_label_folder = './AIWDB_yolov8_sc/train/labels'

    # Перемещаем картинки и файлы меток
    move_images_and_labels(src_image_folder, src_label_folder, dest_image_folder, dest_label_folder)

    # Указываем путь к папке, которую нужно удалить
    folder_to_delete = './train'

    # Удаляем папку
    delete_folder(folder_to_delete)

    # Указываем пути к файлам, которые нужно удалить
    files_to_delete = ['./README.dataset.txt', './README.roboflow.txt']

    # Удаляем файлы
    delete_files(files_to_delete)

    # Указываем путь к папке, которую нужно заархивировать
    folder_to_archive = './AIWDB_yolov8_sc'

    # Указываем путь для сохранения архива
    archive_output_path = './AIWDB_yolov8_sc'

    # Архивируем папку
    archive_folder(folder_to_archive, archive_output_path)