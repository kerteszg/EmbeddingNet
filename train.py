import os
import numpy as np
from embedding_net.model_new import EmbeddingNet, TripletNet
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from embedding_net.datagenerators import ENDataLoader, SimpleDataGenerator, TripletsDataGenerator, SimpleTripletsDataGenerator, SiameseDataGenerator
from embedding_net.utils import parse_params, plot_grapths
from embedding_net.backbones import pretrain_backbone_softmax
import argparse
from tensorflow import keras
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classificator')
    parser.add_argument('config', help='model config file path')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')

    args = parser.parse_args()

    return args

def create_save_folders(params):
    work_dir_path = os.path.join(params['work_dir'], params['project_name'])
    weights_save_path = os.path.join(work_dir_path, 'weights/')
    weights_pretrained_save_path = os.path.join(work_dir_path, 'pretraining_model/weights/')
    encodings_save_path = os.path.join(work_dir_path, 'encodings/')
    plots_save_path = os.path.join(work_dir_path, 'plots/')
    tensorboard_save_path = os.path.join(work_dir_path, 'tf_log/')
    tensorboard_pretrained_save_path = os.path.join(work_dir_path, 'pretraining_model/tf_log/')
    weights_save_file_path = os.path.join(weights_save_path, 'best_' + params['project_name']+'_{epoch:03d}_{loss:03f}' + '.h5')

    os.makedirs(work_dir_path , exist_ok=True)
    os.makedirs(weights_save_path, exist_ok=True)
    os.makedirs(weights_pretrained_save_path, exist_ok=True)
    os.makedirs(encodings_save_path, exist_ok=True)
    os.makedirs(plots_save_path, exist_ok=True)
    os.makedirs(tensorboard_pretrained_save_path, exist_ok=True)

    return tensorboard_save_path, weights_save_file_path, plots_save_path

def main():
    config = tf.ConfigProto(
        device_count={'GPU': 1},
        intra_op_parallelism_threads=1,
        allow_soft_placement=True
    )

    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 1

    session = tf.Session(config=config)

    keras.backend.set_session(session)

    args = parse_args()
    cfg_params = parse_params(args.config)
    params_train = cfg_params['train']
    params_dataloader = cfg_params['dataloader']
    params_generator = cfg_params['generator']

    tensorboard_save_path, weights_save_file_path, plots_save_path = create_save_folders(cfg_params['save_paths'])


    cfg_params['save_paths']
    work_dir_path = os.path.join(cfg_params['save_paths']['work_dir'],
                                 cfg_params['save_paths']['project_name'])
    weights_save_path = os.path.join(work_dir_path, 'weights/')
    

    initial_lr = params_train['learning_rate']
    decay_factor = params_train['decay_factor']
    step_size = params_train['step_size']

    if params_dataloader['validate']:
        callback_monitor = 'val_loss'
    else:
        callback_monitor = 'loss'

    callbacks = [
        LearningRateScheduler(lambda x: initial_lr *
                              decay_factor ** np.floor(x/step_size)),
        ReduceLROnPlateau(monitor=callback_monitor, factor=0.1,
                          patience=4, verbose=1),
        EarlyStopping(monitor=callback_monitor,
                      patience=10, 
                      verbose=1),
        # TensorBoard(log_dir=tensorboard_save_path),
        ModelCheckpoint(filepath=weights_save_file_path,
                        monitor=callback_monitor, 
                        save_best_only=True,
                        verbose=1)
    ]

    data_loader = ENDataLoader(**params_dataloader)
    model = TripletNet(cfg_params, training=True)
    if args.resume_from is not None:
        model.load_model(args.resume_from)
        
    model.load_model('work_dirs/deepfake_efn_b3/weights/best_deepfake_efn_b3_001_0.430115.h5')

    if 'softmax' in cfg_params:
        params_softmax = cfg_params['softmax']
        params_save_paths = cfg_params['save_paths']
        pretrain_backbone_softmax(model.backbone_model, 
                                  data_loader, 
                                  params_softmax,  
                                  params_save_paths)

     # def _create_generators(self):
    #     self.train_generator = SiameseDataGenerator(class_files_paths=self.data_loader.train_data,
    #                                            class_names=self.data_loader.class_names,
    #                                            **self.params_generator)
    #     if self.data_loader.validate:
    #         self.val_generator = SiameseDataGenerator(class_files_paths=self.data_loader.val_data,
    #                                            class_names=self.data_loader.class_names,
    #                                            **self.params_generator)

    # train_generator = SimpleTripletsDataGenerator(class_files_paths=data_loader.train_data,
    #                                         class_names=data_loader.class_names,
    #                                         **params_generator)
    # checkpoints_load_name = 'work_dirs/bengali_efn_b5/pretraining_model/weights/best_efficientnet-b5.hdf5'
    # model.base_model.load_weights(checkpoints_load_name, by_name=True)
    train_generator = TripletsDataGenerator(embedding_model=model.base_model,
                                            class_files_paths=data_loader.train_data,
                                            class_names=data_loader.class_names,
                                            **params_generator)
    

    if data_loader.validate:
        val_generator = SimpleTripletsDataGenerator(data_loader.val_data,
                                               data_loader.class_names,
                                               **params_generator)
    else:
        val_generator = None

    history = model.model.fit_generator(train_generator,
                                           validation_data=val_generator,  
                                           epochs=params_train['n_epochs'], 
                                           callbacks=callbacks,
                                           verbose=1,
                                           use_multiprocessing=False)

    # encoded_training_data = model.generate_encodings(data_loader,
                        #  max_n_samples=50000000000000000, shuffle=False)
    # model.save_encodings(encoded_training_data)

    if params_train['plot_history']:
        plot_grapths(history, plots_save_path)

if __name__ == '__main__':
    main()
