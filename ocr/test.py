import numpy as np
import time
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
import configparser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

from Loss import Loss_CTC
from Metrics import Plates_recognized, Symbols_recognized
from DataGenerator import DataLoader
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass

characters = ['_','1','2','3','4','5','6','7','8','9','0']
def num_to_char(indxs):
    s = ""
    for indx in indxs:
        s += characters[indx]
    return s
#------------------------------------------------------------------------

# class DataLoader(tf.keras.utils.Sequence):
#     def __init__(self, path_data='/data/datasets/ds.video/yakovlev/OCR_car_plate/autoriaNumberplateOcrRu-2021-09-01/train', im_size=[200, 50], batch_size=16,
#                  shuffle=True, char_2_num=None, add_gaus_noise_scale=0):
#         self.im_size = im_size
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.path_data = path_data
#         # self.aug_gausian_nois = iaa.AdditiveGaussianNoise(scale=add_gaus_noise_scale * 255, per_channel=True)
#
#         self.characters = ['-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P',
#                       'C', 'T', 'Y', 'X']  # Все английские, как в датасете 22+1 символа
#         # with tf.device('/device:cpu:0'):
#         # self.    def char_to_num(self, string_plate):
#         #         indxs = []
#         #         for char in string_plate:
#         #             for i, char_in_dict in enumerate(self.characters):
#         #                 if char == char_in_dict:
#         #                     indxs.append(i)
#         #         return np.array(indxs) = tf.keras.layers.StringLookup(vocabulary=list(characters), mask_token=None)
#
#         self.names_img = []
#         for name_img in os.listdir(path_data):
#             self.names_img.append(os.path.join(path_data, name_img))
#         self.indexes = np.arange(len(self.names_img))
#
#     def char_to_num(self, string_plate):
#         indxs = []
#         for char in string_plate:
#             for i, char_in_dict in enumerate(self.characters):
#                 if char == char_in_dict:
#                     indxs.append(i)
#         return np.array(indxs)
#
#     def __getsample__(self, idx):
#         img_path = self.names_img[idx]
#         image = cv2.imread(img_path)
#         # image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, (self.im_size[1], self.im_size[0]))  #width, height
#         image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#         name_img = img_path.split(os.path.sep)[-1].split(".png")[0]
#         name_img = name_img.split(os.path.sep)[-1].split("_")[0]
#         if len(name_img) == 8:
#             label = (name_img + '-')
#         elif len(name_img) == 9:
#             label = (name_img)
#         else:
#             print("Непредвиденное имя файла: ", '"', name_img, '"')
#             exit()
#         # label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
#         label = self.char_to_num(label)
#
#         return image, label
#
#     def __getitem__(self, idx):
#         start_ind = idx * self.batch_size
#         end_ind = (idx + 1) * self.batch_size
#         if end_ind >= len(self.indexes):
#             indexes = self.indexes[start_ind:]
#         else:
#             indexes = self.indexes[start_ind: end_ind]
#         imgs = np.zeros((len(indexes), self.im_size[1], self.im_size[0], 3), dtype=np.uint8)
#         labels = np.zeros((len(indexes), 9), dtype=np.int64)
#         for sample_ind, ind in enumerate(indexes):
#             imgs[sample_ind, ...], labels[sample_ind, ...] = self.__getsample__(ind)
#         # imgs = self.aug_gausian_nois(images=imgs)
#         imgs = imgs.astype(np.float32) / 255.
#         labels = labels.astype(np.int64)
#         return imgs, labels
#
#     def __len__(self):
#         return np.ceil(len(self.indexes) / self.batch_size).astype(int)
#
#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indexes)


# parh_data = r"F:\Магистратура\NIR\OCR\example_pytesseract\Data\test_augmentation_small_aug"
# parh_data = r"F:\Магистратура\NIR\OCR\example_pytesseract\Data\test_augmentation"
# parh_data = r"F:\Магистратура\NIR\OCR\Datasets\cleared__autoriaNumberplateOcrRu-2021-09-01\autoriaNumberplateOcrRu-2021-09-01\test\img"
# parh_data = r"F:\Магистратура\NIR\OCR\example_pytesseract\Data\augmentation_images"

def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len,
                                           greedy=True)[0][0][:, :8]
        # results_2 = tf.sparse.to_dense(tf.nn.ctc_greedy_decoder(tf.transpose(pred, [1, 0, 2]), input_len, merge_repeated=False, blank_index=None)[0][0])[:, :8]
        output_text = []
        for res in results:
            res = num_to_char(res)
            output_text.append(res)
        return output_text, results


def recovery_char(wagon_plate):
    weights = [2, 1, 2, 1, 2, 1, 2, 1]

    def check_summ(chars):
        summ = 0
        for w, char in zip(weights[0:-1], chars[0:-1]):
            summ += np.sum([int(ch) for ch in str(int(w) * int(char))])
        nearest_decade = (int(str(summ)[0]) + 1) * 10
        if wagon_plate[-1] == str(nearest_decade - summ):
            err = False
        else:
            err = True

        return str(nearest_decade - summ), err

    count_err = wagon_plate.count("_")
    count_err += wagon_plate.count("_")
    if count_err == 0:
        _, err = check_summ(wagon_plate)
        return wagon_plate, err
    elif count_err > 1:
        return wagon_plate, True
    elif wagon_plate[-1] == "_" or wagon_plate[-1] == "_":
        hash_summ, _ = check_summ(wagon_plate)
        wagon_plate[-1] = hash_summ
        return wagon_plate, False
    else:
        for i in range(10):
            wg_plate = wagon_plate
            wg_plate.replace("_", str(i))
            wg_plate.replace("_", str(i))
            hash_summ, _ = check_summ(wg_plate)
            if wg_plate[-1] == hash_summ:
                wagon_plate[wagon_plate.find("_")] = str(i)
                wagon_plate[wagon_plate.find("_")] = str(i)
                return wagon_plate, False

        return wagon_plate, True

def test(config):
    shape_inp_img = tuple([np.int32(i) for i in (config["shape"].split(','))])
    test_dl = DataLoader(Path(config["test_data_dir"]),
                        im_size=shape_inp_img,
                        batch_size=config.getint("batch_size"),
                        shuffle=False,
                        augmentation=False)

    symbols_rec = Symbols_recognized()
    plates_rec = Plates_recognized()
    prediction_model = tf.keras.models.load_model(filepath = config["model_path"],
                                                  custom_objects = {"Loss_CTC": Loss_CTC,
                                                                    "Symbols_recognized": symbols_rec,
                                                                    "Plates_recognized": plates_rec})
    prediction_model.summary()

    all_orig_texts = []
    all_pred_texts = []
    for index in range(len(test_dl)):
        batch_images, batch_labels = test_dl.__getitem__(index)
        t_start = time.time()
        preds = prediction_model.predict(batch_images)
        pred_texts, res = decode_batch_predictions(preds)
        print(time.time() - t_start)
        all_pred_texts.extend(pred_texts)
        orig_texts = []
        for label in batch_labels:
            label = num_to_char(label)
            orig_texts.append(label)
            all_orig_texts.append(label)
    for plate in all_pred_texts:
        plate, err = recovery_char(plate)

    with open("Result_test_0753.txt", 'w', encoding="utf-8" ) as result_file:
        errors_chars, errors_plates, correct_err = 0, 0, 0
        for orig, pred in zip(all_orig_texts, all_pred_texts):
            rec_plate, err = recovery_char(pred)
            result_file.write(orig + ' ' + rec_plate + "  errore: " + str(err) + '\n')
            error_this_car_plate = False
            for i, char in enumerate(orig):
                if char != rec_plate[i]:
                    error_this_car_plate = True
                    errors_chars += 1
            if error_this_car_plate:
                errors_plates += 1
            if error_this_car_plate and not(err):
                correct_err += 1


        # Ошибки в процентном выражении
        percent_errors_chars = (errors_chars / (len(all_orig_texts) * 8) )
        percent_errors_plates = (errors_plates / len(all_orig_texts))
        percept_correct_err = (correct_err / len(all_orig_texts))
        accuracy_chars_recognition = 1 - percent_errors_chars
        accuracy_plates_recognition = 1 - percent_errors_plates
        result_file.write('count_error_chars: ' + str(errors_chars) +'  |  '
                          + str(percent_errors_chars) + '\n')
        result_file.write('count_error_car_plate: ' + str(errors_plates) +'  |  '
                          + str(percent_errors_plates) + '\n')
        result_file.write('correct_err: ' + str(correct_err) + '  |  '
                          + str(percept_correct_err) + '\n')

        result_file.write('\n' + 'accuracy_chars_recognition: ' + str(accuracy_chars_recognition) + '\n')
        result_file.write('accuracy_plates_recognition: ' + str(accuracy_plates_recognition) + '\n')


#     _, ax = plt.subplots(4, 4, figsize=(15, 5))
#     for i in range(len(pred_texts)):
#         img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
#         img = img.T
#         title = f"Prediction: {pred_texts[i]}"
#         ax[i // 4, i % 4].imshow(img, cmap="gray")
#         ax[i // 4, i % 4].set_title(title)
#         ax[i // 4, i % 4].axis("off")
# plt.show()

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass
    config = configparser.ConfigParser()
    config.read("Config.ini")
    test(config["Test"])
