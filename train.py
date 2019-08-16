import os
import numpy as np
from model import SiameseNet
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


n_epochs = 5
n_steps_per_epoch = 6
batch_size = 1
val_steps = 100

# model = SiameseNet('configs/road_signs.yml')
model = SiameseNet('configs/plates.yml')


initial_lr = 1e-4
decay_factor = 0.99
step_size = 1

callbacks = [
    LearningRateScheduler(lambda x: initial_lr *
                          decay_factor ** np.floor(x/step_size)),
    EarlyStopping(patience=50, verbose=1),
    TensorBoard(log_dir=model.tensorboard_log_path),
    ModelCheckpoint(filepath=os.path.join(model.weights_save_path, 'best_model.h5'),
                    verbose=1, monitor='val_loss', save_best_only=True)
]

model.train_generator(steps_per_epoch=n_steps_per_epoch, callbacks=callbacks,
                      val_steps=val_steps, epochs=n_epochs)


model.generate_encodings()
# model.load_encodings('encodings/encodings.pkl')

model_accuracy = model.calculate_prediction_accuracy()
print('Model accuracy on validation set: {}'.format(model_accuracy))