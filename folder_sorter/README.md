# Folder_sorter  
 Модуль для предобработки и создания конечного датасета

 _Данный модуль формирует пропорциональные классам выборки train/test/valid с помощью библиотеки scikit-learn_
 _Автоматически формирует датасет в формате *YOLOv8.zip_

## Использование

1. Необходимо выгрузить из *Roboflow* и поместить в корневую папку модуля folder_sorter датасет, в формате *tensorflow.zip без аугментации.  

2. Необходимо выгрузить и поместить из *Roboflow* датасет с аугментацией без разбиения на  
выборки train/test/val в формате *yolov8.zip, поместить в корневую папку модуля folder_sorter датасет.  

3. Запускаем модуль:

```
python folder_sorter/main.py
```

_Полученный датасет AIWDB_yolov8_sc далее используем для работы с моделью YOLOv8_