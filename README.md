# Project for hackathon 2023 season AI Цифровой Прорыв.
## Команда "Исходный Кот :) "

Наша команда представляет алгоритм, предназначенный для обнаружения и распознавания номеров железнодорожных вагонов для автоматической фиксации проезда вагонов по динамическим весам.

!⁠[dinamic weight]​(https://github.com/Ophelials/Src_cat/blob/1088118b5335106791c9cf74f4f2237ab602bb21/images/results.png)
Данный алгоритм позволит максимально автоматизировать и оптимизировать процесс динамического взвешивания.

### Используемый стек: 
- Python;
- OpenCV; 
- Tensorflow;
- Pytorch;
- YOLOv8x;
- ResNetRS50.

### Конечное решение состоит из двух частей: 
1) Детекция номера на изображении (определение координат ограничивающих рамок).
2) Распознавание символов в вырезанном изображении номера.

Разделение на две части позволило решить подзадачи независимо друг от друга, что расширило выбор используемых технологий.


Детекция номера (локализация) производится с помощью одной из лучших в своём классе нейронной сетью (НС) YOLOv8x.

https://github.com/Ophelials/Src_cat/blob/main/images/yolov8_architecture.jpg

Распознавание символов номера осуществляется за счёт кастомной нейронной сети построенной по типу НС для решения задачи OCR (optical character recognition(оптическое распознавание символов)). Она состоит из нейронной сети ResNet50, которая формирует признаки объектов изображённых на изображении, и элементов LSTM (элементы долгой и краткосрочной памяти), которые формируют по этим признакам конечную последовательность символов номера.

## Object Detection

В качесттве модели детектирования была взята предобученная модель Yolov8x, что позволило значительно увеличить точность предсказаний, несмотря на малое количество данных.
Обучение 300 обучающих эпох производилось на GPU.
Конечные полученные метрики представлены на рисунке:

!⁠[yolo_results]​(https://github.com/Ophelials/Src_cat/blob/main/images/results.png)

Можно заметить что модель с высокой точностью верно предсказывает нахождение номеров железнодорожных вагонов.

!⁠[yolo_results2]​(images/results.jpg)

После обучения модели детектирования по полученным из модели данным номера на фотографиях обрезаются с помощью инструментов библиотеки OpenCV и подаются в модель OCR.

## Optical Character Recognition
