# Система видеодетекции объектов нестационарной незаконной торговли

Эта программа разработана на языке Python 3.11 и предназначена для автоматической передачи данных о выявленных объектах в правоохранительные органы, структуры, ответственные за надзор в сфере защиты прав потребителей, администрации муниципальных образований. 

## Установка

Для работы программы необходимо установить следующие библиотеки:

- ultralytics
- fastapi
- bootstrap-py
- python-ffmpeg
- imagehash

Выполните следующую команду для установки библиотек:
>     $pip install ultralytics fastapi bootstrap-py python-ffmpeg imagehash

## Функциональность сервиса
- Детекции объектов нестационарной незаконной торговли на подгружаемых видеозаписях
- Оперативное выявление точек незаконной торговли в режиме реального времени на основе типа объекта (установленный или с возможностью передвижения), его размера и местонахождения
- Интеграция с инфраструктурой системы видеонаблюдения (городской, частной охранной и мультикамерной)

## Состав репозитория

### Запуск интерфейса (Backend+Frontend)

## Возможность дообучения моделей

## Требования
### Минимальные	
- Операционная система	- Windows 8, Mac OS 10.11 (El Capitan)
- Процессор	- Intel Core i3 или аналогичный
- Оперативная память	- 2 Гб
- Видеокарта	- Объемом видеопамяти не менее 500 Мб (GeForce 7300 GT, Intel HD Graphics 620 и выше)
### Рекомендованные
- Операционная система	- Windows 10/11 Mac OS 10.13 (Ventura)
- Процессор	- Двухъядерный Intel Core i5 или аналогичный
- Оперативная память	- 4 Гб
- Видеокарта	- Объемом видеопамяти не менее 1 Гб (Intel HD Graphics 610, NVIDIA GeForce 930MX и выше)
- 
## Подготовка набора данных

- Использование модуля предобработки
- Выделение внутри точек нестационарной незаконной торговли специфичные классы по типам объектов
- Использование разметки "полигонами"
- Разделение выборки на train, val, test
- Добавление в набор данных, не содержавших объекты целевых классов

## Лицензирование

Этот проект распространяется без лицензии.

Не стесняйтесь вносить свои предложения по улучшению или доработке фукциональности сервиса

## Команда проекта

__AI Wizardry:__ Роберт Халиуллин, Александр Бульбенков, Даниил Бураков, Мария Тимонина, Михаил Попов
