from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import configparser
import numpy as np

from DataGenerator import DataLoader
from Loss import Loss_CTC
from Metrics import Plates_recognized, Symbols_recognized
from Build_model import build_model

def MyLRSchedule(epoch, lr):
    if not(epoch % 100) and epoch != 0:
        lr = lr * 0.5
    return lr

def train(config):
    images_tr = sorted(list(map(str, list(Path(config["train_data_dir"]).glob("*.png")))))
    #  10+1
    characters = ['_','1','2','3','4','5','6','7','8','9','0']

    print("Number of images found: ", len(images_tr))
    print("Number of unique characters: ", len(characters))
    print("Characters present: ", characters)

    shape_inp_img = tuple([np.int32(i) for i in (config["shape"].split(','))])
    
    train_dl = DataLoader(Path(config["train_data_dir"]),
                          im_size=shape_inp_img,
                          batch_size=config.getint("batch_size"),
                          shuffle=True,
                          augmentation=True)
    val_dl = DataLoader(Path(config["val_data_dir"]),
                        im_size=shape_inp_img,
                        batch_size=config.getint("batch_size"),
                        shuffle=False,
                        augmentation=False)

    model = build_model(len(characters), shape_inp_img)
    opt = keras.optimizers.Adam(learning_rate=config.getfloat("start_lr"))

    model.compile(optimizer=opt, loss=Loss_CTC, metrics=[Symbols_recognized(), Plates_recognized()])
    model.summary()

    logdir = Path(config["log_dir"]) / datetime.now().strftime(config["save_name_model"] + "__%d_%m_%Y__%H_%M_%S")
    check_point_dir = logdir / "checkpoints"
    Path.mkdir(check_point_dir, parents=True)
    #tf.keras.utils.plot_model(model, to_file=(str(logdir / config["save_name_model"]) + ".png"),
    #                          show_shapes=True,
    #                          show_trainable=True)

    tensorboard_cbk = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=0,
        write_grads=False,
        update_freq='epoch',
        write_graph=True)

    check_point_cbk = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(check_point_dir / "{epoch:04d}--val_plates_recognized-{val_plates_recognized:.3f}.h5"),
        monitor='val_plates_recognized',
        mode='max',
        save_best_only=False,
        save_weights_only=False)

    LRScheduler = tf.keras.callbacks.LearningRateScheduler(MyLRSchedule)
    model.fit(
        train_dl,
        validation_data=val_dl,
        validation_freq=1,
        epochs=config.getint("epochs"),
        callbacks=[LRScheduler, tensorboard_cbk, check_point_cbk],
        max_queue_size=512,
        workers=4,
        use_multiprocessing=False
    )

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass
    config = configparser.ConfigParser()
    config.read("Config.ini")
    train(config["Train"])
