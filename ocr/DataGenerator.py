import tensorflow as tf
import albumentations as A
import numpy as np
import cv2

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, path_data='/Data/train', im_size=[200, 100, 3], batch_size=16,
                 shuffle=True, augmentation=False, char_2_num=None):
        self.im_size = im_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.path_data = path_data
        self.characters = ['_','1','2','3','4','5','6','7','8','9','0']
        self.names_img = []
        self.dataset = []
        for name_img in (self.path_data / "images").glob("*[.png, .jpg, .jpeg, .tiff, .bmp, .gif]"):
            with open(str(self.path_data / "labels" / name_img.stem) + ".txt") as f:
                label = f.readlines()
            if label == []:
                continue
            label = label[0].replace("\n", "").split(" ")
            self.dataset.append({"path_img": (self.path_data / "images" / name_img),
                                 "bboxes": [float(cord) for cord in label[1:]],
                                 "class": int(label[0]),
                                 "num_plate": name_img.stem})
        self.indexes = np.arange(len(self.dataset))
        self.len = len(self.indexes) // batch_size
        if len(self.indexes) % batch_size:
            self.len += 1
        self.on_epoch_end()
        if self.augmentation:
            self.aug_gausian_nois = A.Compose([A.OneOf([
                A.GaussNoise(var_limit=50.0, per_channel=True),
                A.GaussianBlur (blur_limit=(3, 9), sigma_limit=(0, 3)),
                A.ImageCompression (quality_lower=70, quality_upper=90)])], p=0.95)
            self.another_augmentation = A.Compose(A.RandomBrightnessContrast(brightness_limit=(0.8, 1.3),
                                                                             contrast_limit=0.2,
                                                                             brightness_by_max=True,
                                                                             always_apply=False,
                                                                             p=1), p=0.3)
    def char_to_num(self, string_plate):
        indxs = []
        for char in string_plate:
            for i, char_in_dict in enumerate(self.characters):
                if char == char_in_dict:
                    indxs.append(i)
        return np.array(indxs)

    def crop_plate(self, bbox, larget_img, alpha=0.3):
        left_up_x = max(int((bbox[0] - bbox[2] * alpha / 2) * larget_img.shape[1]), 0)
        left_up_y = max(int((bbox[1] - bbox[3] * alpha / 2) * larget_img.shape[0]), 0)
        right_down_x = min(int((bbox[0] + bbox[2] * alpha / 2) * larget_img.shape[1]), larget_img.shape[1])
        right_down_y = min(int((bbox[1] + bbox[3] * alpha / 2) * larget_img.shape[0]), larget_img.shape[0])
        return larget_img[left_up_y : right_down_y, left_up_x : right_down_x, :].copy()

    def __getsample__(self, idx):
        example = self.dataset[idx]
        image = cv2.imread(str(example["path_img"]))
        plate = self.crop_plate(example["bboxes"], image.copy(), alpha=1 + 0.3)

        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
        plate = cv2.rotate(plate, cv2.ROTATE_90_CLOCKWISE)
        plate = cv2.resize(plate, (self.im_size[1], self.im_size[0]))  #width, height

        label = example["num_plate"]
        label = self.char_to_num(label)
        if self.augmentation:
            plate_aug = self.another_augmentation(image=plate)
            # imgs = self.another_augmentation_2(images=imgs)
            plate_aug = self.aug_gausian_nois(image=plate_aug["image"])["image"]
            return plate_aug, label
        else:
            return plate, label

    def __getitem__(self, idx):
        start_ind = idx * self.batch_size
        end_ind = (idx + 1) * self.batch_size
        if end_ind >= len(self.indexes):
            indexes = self.indexes[start_ind:]
        else:
            indexes = self.indexes[start_ind: end_ind]
        imgs = np.zeros((len(indexes), self.im_size[0], self.im_size[1], self.im_size[2]), dtype=np.uint8)
        labels = np.zeros((len(indexes), 8), dtype=np.int64)
        for sample_ind, ind in enumerate(indexes):
            imgs[sample_ind, :, :, :], labels[sample_ind, :] = self.__getsample__(ind)
        imgs = imgs.astype(np.float32) / 255.
        labels = labels.astype(np.int64)
        return imgs, labels

    def __len__(self):
        return self.len

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)