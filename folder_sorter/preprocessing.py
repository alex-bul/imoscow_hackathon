import os
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split


class LabelsConverter:
    def __init__(self, outdir='.'):
        self.outdir = outdir

    def transform_image_labels(self, path_to_labels_txt: str, to: str = 'file'):
        '''
        The function reads image labels txt file line by line
        and converts any bbox to polygon saving modified txt file
        in the stated output directory
        '''
        new_lines = ''
        with open(path_to_labels_txt, 'r') as source_file:
            for line in source_file:
                tokens = line.strip().split(' ')
                class_id = tokens[0]
                coords = list(map(float, tokens[1:]))
                if len(coords) == 4:
                    pn = self.from_xywhn_to_polygonn(coords)
                    new_line = str(class_id)
                    for point in pn:
                        new_line += (' ' + str(point))
                else:
                    new_line = line
                new_lines += (new_line + '\n')

        if to == 'file':
            new_labels_path = os.path.basename(path_to_labels_txt)
            with open(new_labels_path, 'w') as target_file:
                target_file.write(new_lines)
        else:
            print(new_lines)

    @staticmethod
    def from_xyxy_to_xywhn(img_size: tuple[int], box: list[float]):
        '''
        input: (img_w, img_h), (x1, y1, x2, y2) - top left and bottom right
        output: [xn, yn, wn, hn] - bbox center, width, height, all normalized
        '''
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        xn = x / img_size[0]
        wn = w / img_size[0]
        yn = y / img_size[1]
        hn = h / img_size[1]
        return [xn, yn, wn, hn]

    @staticmethod
    def from_xywhn_to_polygonn(xywhn: list[float]):
        '''
        input: [xn, yn, wn, hn] - bbox center, width, height, all normalized
        output: [x1n, y1n, x2n, y2n, x3n, y3n, x4n, y4n]
        '''
        xn, yn, wn, hn = xywhn
        # top left
        x1n = xn - wn / 2.0
        y1n = yn - hn / 2.0
        # top right
        x2n = xn + wn / 2.0
        y2n = yn - hn / 2.0
        # bottom right
        x3n = xn + wn / 2.0
        y3n = yn + hn / 2.0
        # bottom left
        x4n = xn - wn / 2.0
        y4n = yn + hn / 2.0
        return [x1n, y1n, x2n, y2n, x3n, y3n, x4n, y4n]

    @staticmethod
    def from_bbox_polygonn_to_xywh(polygonn: tuple[float]):
        '''
        input: (x1n, y1n, x2n, y2n, x3n, y3n, x4n, y4n)
        output: (xn, yn, wn, hn) - bbox center, width, height, all normalized
        '''
        x1n, y1n, x2n, y2n, x3n, y3n, x4n, y4n = polygonn
        xn = (x1n + x2n) / 2.0
        wn = (x2n - x1n)

        yn = (y1n + y4n) / 2.0
        hn = (y4n - y1n)
        return [xn, yn, wn, hn]


class DataPreprocessor:
    '''Разделитель тренировочного датасета с двумя и более классами на пропорциональные выборки train, test и valid'''
    def __init__(self, val_size: float = 0.15, test_size: float = 0.15):
        self.val_size = val_size
        self.test_size = test_size

    def split_roboflow_dataset(self, annotations_csv: str, output_dir: str):
        if not os.path.exists(annotations_csv):
            raise "Bad path: Annotations csv not found"

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        df_annots = pd.read_csv(annotations_csv)

        for field in ['filename', 'class']:
            if field not in df_annots.columns:
                raise f"Bad content: '{field}' column in annotations csv not found"

        file_bestclass = df_annots.groupby('filename')['class'].agg(list).apply(
            lambda s: Counter(s).most_common()[0][0]
        )

        matching_table = pd.DataFrame(file_bestclass).reset_index()
        matching_table = matching_table.rename({'class': 'major_class'}, axis=1)

        # split train filenames from val and test
        files_train, files_valtest, _, class_valtest = train_test_split(
            matching_table, matching_table['major_class'],
            test_size=(self.val_size + self.test_size),
            random_state=222,
            stratify=matching_table['major_class']
        )

        # split test filenames from val
        files_val, files_test, y_val, y_test = train_test_split(
            files_valtest, class_valtest,
            test_size=self.test_size / (self.val_size + self.test_size),
            random_state=222,
            stratify=class_valtest
        )

        files_train['split'] = 'train'
        files_val['split'] = 'val'
        files_test['split'] = 'test'

        files_merged = pd.concat([files_train, files_val, files_test]).set_index('filename')

        df_annots['split'] = df_annots['filename'].apply(
            lambda f: files_merged.loc[f, 'split']
        )

        if (df_annots.groupby('filename')['split'].agg('nunique') != 1).sum() != 0:
            raise "Split not successful: objects of one filename have different split directories"

        df_annots[df_annots['split'] == 'train'].to_csv(os.path.join(output_dir, 'train_split.csv'))
        df_annots[df_annots['split'] == 'val'].to_csv(os.path.join(output_dir, 'val_split.csv'))
        df_annots[df_annots['split'] == 'test'].to_csv(os.path.join(output_dir,'test_split.csv'))










