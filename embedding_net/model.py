import os
import numpy as np
import keras.backend as K
import cv2
import random
import keras
from keras.models import Model, load_model
from keras import optimizers
from keras.layers import Dense, Input, Lambda, concatenate
import pickle
from .utils import load_encodings
from .backbones import get_backbone
from . import losses_and_accuracies as lac
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy.random as rng
from sklearn.utils import shuffle

# TODO
# [] - implement magnet loss
# [] - finalize settings with l1 and l2 losses


class EmbeddingNet:
    """
    SiameseNet for image classification
    distance_type = 'l1' -> l1_loss
    distance_type = 'l2' -> l2_loss

    mode = 'siamese' -> Siamese network
    mode = 'triplet' -> Triplen network
    """

    def __init__(self,  cfg_params, training = True):
        self.input_shape = cfg_params['input_shape']
        self.encodings_len = cfg_params['encodings_len']
        self.backbone = cfg_params['backbone']
        self.backbone_weights = cfg_params['backbone_weights']
        self.distance_type = cfg_params['distance_type']
        self.mode = cfg_params['mode']
        self.optimizer = cfg_params['optimizer']
        self.freeze_backbone = cfg_params['freeze_backbone']
        self.data_loader = cfg_params['loader']
        self.embeddings_normalization = cfg_params['embeddings_normalization']
        self.margin = cfg_params['margin']
        self.cfg_params = cfg_params

        self.model = []
        self.base_model = []
        self.backbone_model = []

        if self.mode == 'siamese':
            self._create_model_siamese(cfg_params['verbose'])
        elif self.mode == 'triplet':
            self._create_model_triplet(cfg_params['verbose'])
        else:
            self._create_base_model()

        self.encoded_training_data = {}

        if cfg_params['softmax_pretraining'] and training:
            self.pretrain_backbone_softmax()

    def pretrain_backbone_softmax(self):
        input_shape = self.cfg_params['input_shape']
        dataset_path = self.cfg_params['dataset_path']
        n_classes = self.data_loader.n_classes['train']
        if 'softmax_is_binary' in self.cfg_params:
            is_binary = self.cfg_params['softmax_is_binary']
        else: 
            is_binary = False

        if len(self.backbone_model.output.shape) == 4:
            x = keras.layers.GlobalAveragePooling2D()(self.backbone_model.output)
        else:
            x = self.backbone_model.output
        if is_binary:
            output = keras.layers.Dense(1, activation='softmax')(x)
        else:
            output = keras.layers.Dense(n_classes, activation='softmax')(x)
        model = keras.models.Model(inputs=[self.backbone_model.input], outputs=[output])

        # train
        mloss = 'binary_crossentropy' if is_binary else 'categorical_crossentropy'
        model.compile(optimizer='Adam',
                    loss=mloss, metrics=['accuracy'])

        batch_size_train = self.cfg_params['softmax_batch_size_train']
        batch_size_val = self.cfg_params['softmax_batch_size_val']
        val_steps = self.cfg_params['softmax_val_steps']
        steps_per_epoch = self.cfg_params['softmax_steps_per_epoch']
        epochs = self.cfg_params['softmax_epochs']

        train_generator = self.data_loader.generate(batch_size_train,is_binary=is_binary, mode='simple', s="train")
        if 'val' in self.data_loader.data_subsets and self.cfg_params['to_validate']:
            val_generator = self.data_loader.generate(batch_size_val,is_binary=is_binary, mode='simple', s="val")
            checkpoint_callback_monitor = 'val_loss'
        else:
            val_generator = None
            checkpoint_callback_monitor = 'loss'

        tensorboard_save_path = os.path.join(
            self.cfg_params['work_dir'], 'tf_log/pretraining_model/')
        weights_save_file = os.path.join(
            self.cfg_params['work_dir'], 
            'weights/pretraining_model/',
            self.cfg_params['model_save_name'])

        initial_lr = self.cfg_params['learning_rate']
        decay_factor = self.cfg_params['decay_factor']
        step_size = self.cfg_params['step_size']

        callbacks = [
            LearningRateScheduler(lambda x: initial_lr *
                                decay_factor ** np.floor(x/step_size)),
            ReduceLROnPlateau(monitor=checkpoint_callback_monitor, factor=0.1,
                            patience=20, verbose=1),
            EarlyStopping(monitor=checkpoint_callback_monitor,
                          patience=10, 
                          verbose=1, 
                          restore_best_weights=True),
            TensorBoard(log_dir=tensorboard_save_path),
            ModelCheckpoint(filepath=weights_save_file,
                            verbose=1, 
                            monitor=checkpoint_callback_monitor, 
                            save_best_only=True)
        ]

        history = model.fit_generator(train_generator,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=val_generator,
                                    validation_steps=val_steps,
                                    callbacks=callbacks)

    def _create_base_model(self):
        self.base_model, self.backbone_model = get_backbone(input_shape=self.input_shape,
                                                            encodings_len=self.encodings_len,
                                                            backbone_type=self.backbone,
                                                            embeddings_normalization=self.embeddings_normalization,
                                                            backbone_weights=self.backbone_weights,
                                                            freeze_backbone=self.freeze_backbone)

    def _create_model_siamese(self, verbose=True):

        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)

        self._create_base_model()
        self.base_model._make_predict_function()

        image_encoding_1 = self.base_model(input_image_1)
        image_encoding_2 = self.base_model(input_image_2)

        if self.distance_type == 'l1':
            L1_layer = Lambda(
                lambda tensors: K.abs(tensors[0] - tensors[1]))
            distance = L1_layer([image_encoding_1, image_encoding_2])

            prediction = Dense(units=1, activation='sigmoid')(distance)
            metric = 'binary_accuracy'

        elif self.distance_type == 'l2':

            L2_layer = Lambda(
                lambda tensors: K.sqrt(K.maximum(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True), K.epsilon())))
            distance = L2_layer([image_encoding_1, image_encoding_2])

            prediction = distance
            metric = lac.accuracy

        self.model = Model(
            inputs=[input_image_1, input_image_2], outputs=prediction)

        if verbose:
            print('Base model summary')
            self.base_model.summary()

            print('Whole model summary')
            self.model.summary()

        self.model.compile(loss=lac.contrastive_loss, metrics=[metric],
                           optimizer=self.optimizer)

    def _create_model_triplet(self, verbose=True):
        input_image_a = Input(self.input_shape)
        input_image_p = Input(self.input_shape)
        input_image_n = Input(self.input_shape)

        self._create_base_model()
        self.base_model._make_predict_function()
        image_encoding_a = self.base_model(input_image_a)
        image_encoding_p = self.base_model(input_image_p)
        image_encoding_n = self.base_model(input_image_n)

        merged_vector = concatenate([image_encoding_a, image_encoding_p, image_encoding_n],
                                    axis=-1, name='merged_layer')
        self.model = Model(inputs=[input_image_a, input_image_p, input_image_n],
                           outputs=merged_vector)

        if verbose:
            print('Base model summary')
            self.base_model.summary()

            print('Whole model summary')
            self.model.summary()

        self.model.compile(loss=lac.triplet_loss(
            self.margin), optimizer=self.optimizer)

    def train_on_batch(self, batch_size=8, s="train"):
        generator = self.data_loader.generate(batch_size, s=s)
        pairs, targets = next(generator)
        train_loss, train_accuracy = self.model.train_on_batch(
            pairs, targets)
        return train_loss, train_accuracy

    def validate_on_batch(self, batch_size=8, s="val"):
        generator = self.data_loader.generate(batch_size, s=s)
        pairs, targets = next(generator)
        val_loss, val_accuracy = self.model.test_on_batch(
            pairs, targets)
        return val_loss, val_accuracy

    def train_generator(self, 
                        steps_per_epoch, 
                        epochs, 
                        callbacks=[], 
                        val_steps=100,  
                        batch_size=8, 
                        verbose=1):

        train_generator = self.data_loader.generate(
            batch_size, mode=self.mode, s="train")
        
        if 'val' in self.data_loader.data_subsets and self.cfg_params['to_validate']:
            val_generator = self.data_loader.generate(
                batch_size, mode=self.mode, s="val")
        else:
            val_generator = None

        history = self.model.fit_generator(train_generator, 
                                           steps_per_epoch=steps_per_epoch, 
                                           epochs=epochs,
                                           verbose=verbose, 
                                           validation_data=val_generator, 
                                           validation_steps=val_steps, 
                                           callbacks=callbacks)

        return history

    def train_generator_mining(self,
                               steps_per_epoch,
                               epochs, 
                               callbacks=[],
                               val_steps=100,
                               n_classes=4,
                               n_samples=4,
                               val_batch=8,
                               negative_selection_mode='hardest',
                               verbose=1):

        train_generator = self.data_loader.generate_mining(
            self.base_model, n_classes, n_samples, margin=self.margin, negative_selection_mode=negative_selection_mode, s="train")
        
        if 'val' in self.data_loader.data_subsets and self.cfg_params['to_validate']:
            val_generator = self.data_loader.generate(
                val_batch, mode=self.mode, s="val")
        else:
            val_generator = None

        history = self.model.fit_generator(train_generator,
                                           steps_per_epoch=steps_per_epoch,
                                           epochs=epochs,
                                           verbose=verbose,
                                           validation_data=val_generator,
                                           validation_steps=val_steps,
                                           callbacks=callbacks)
        return history

    def validate(self, number_of_comparisons=100, batch_size=4, s="val"):
        generator = self.data_loader.generate(batch_size, s=s)
        val_accuracies_it = []
        val_losses_it = []
        for _ in range(number_of_comparisons):
            pairs, targets = next(generator)

            val_loss_it, val_accuracy_it = self.model.test_on_batch(
                pairs, targets)
            val_accuracies_it.append(val_accuracy_it)
            val_losses_it.append(val_loss_it)
        val_loss_epoch = sum(val_losses_it) / len(val_losses_it)
        val_accuracy_epoch = sum(
            val_accuracies_it) / len(val_accuracies_it)
        return val_loss_epoch, val_accuracy_epoch


    def _generate_encoding(self, img_path):
        img = self.data_loader.get_image(img_path)
        if img is None:
            return None
        encoding = self.base_model.predict(np.expand_dims(img, axis=0))
        return encoding


    def generate_encodings(self, save_file_name='encodings.pkl', only_centers=False, max_num_samples_of_each_class=10, knn_k=1, shuffle=True):
        data_paths, data_labels, data_encodings = [], [], []
        classes_counter = {}
        classes_encodings = {}
        k_val = 1 if only_centers else knn_k

        if shuffle:
            c = list(zip(
                self.data_loader.images_paths['train'], self.data_loader.images_labels['train']))
            random.shuffle(c)
            self.data_loader.images_paths['train'], self.data_loader.images_labels['train'] = zip(
                *c)

        for img_path, img_label in zip(self.data_loader.images_paths['train'],
                                       self.data_loader.images_labels['train']):
            if only_centers:
                if img_label not in classes_encodings:
                    classes_encodings[img_label] = []
            else:
                if img_label not in classes_counter:
                    classes_counter[img_label] = 0
            if classes_counter[img_label] < max_num_samples_of_each_class:
                encod = self._generate_encoding(img_path)
                
                if encod is not None:
                    if only_centers:
                        classes_encodings[img_label].append(encod)
                    else:
                        data_paths.append(img_path)
                        data_labels.append(img_label)
                        data_encodings.append(encod)
                        classes_counter[img_label] += 1
        if only_centers:
            for class_i, encodings_i in classes_encodings.items():
                encodings_i_np = np.array(encodings_i)
                class_encoding = np.mean(encodings_i_np, axis = 0)
                data_encodings.append(class_encoding)
                data_labels.append(class_i)
        self.encoded_training_data['paths'] = data_paths
        self.encoded_training_data['labels'] = data_labels
        self.encoded_training_data['encodings'] = np.squeeze(
            np.array(data_encodings))
        self.encoded_training_data['knn_classifier'] = KNeighborsClassifier(
            n_neighbors=k_val)
        self.encoded_training_data['knn_classifier'].fit(self.encoded_training_data['encodings'],
                                                         self.encoded_training_data['labels'])
        with open(save_file_name, "wb") as f:
            pickle.dump(self.encoded_training_data, f)

    def load_encodings(self, path_to_encodings):
        self.encoded_training_data = load_encodings(path_to_encodings)

    def load_model(self, file_path, mode='triplet', _compile=True):
        from keras_radam import RAdam
        self.model = load_model(file_path,
                                custom_objects={'contrastive_loss': lac.contrastive_loss,
                                                'accuracy': lac.accuracy,
                                                'loss_function': lac.triplet_loss(self.margin),
                                                'RAdam': RAdam},
                                compile=_compile)
        if not _compile:
            print(self.model.summary())
            self.model.compile(loss=lac.contrastive_loss, optimizer=self.optimizer)
        self.input_shape = list(self.model.inputs[0].shape[1:])
        if mode == 'triplet':
            self.base_model = Model(inputs=[self.model.layers[3].get_input_at(0)],
                                    outputs=[self.model.layers[3].layers[-1].output])
        elif mode == 'siamese':
            self.base_model = Model(inputs=[self.model.layers[2].get_input_at(0)],
                                    outputs=[self.model.layers[2].layers[-1].output])
        else:
            raise Exception('unknown mode')
        self.base_model._make_predict_function()

    def calculate_distances(self, encoding):
        training_encodings = self.encoded_training_data['encodings']
        return np.sqrt(
            np.sum((training_encodings - np.array(encoding))**2, axis=1))

    def predict(self, image):
        if type(image) is str:
            img = cv2.imread(image)
        else:
            img = image
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
        encoding = self.base_model.predict(np.expand_dims(img, axis=0))
        distances = self.calculate_distances(encoding)
        max_element = np.argmin(distances)
        predicted_label = self.encoded_training_data['labels'][max_element]
        return predicted_label

    def predict_knn(self, image, with_top5=False):
        if type(image) is str:
            img = cv2.imread(image)
        else:
            img = image
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))

        encoding = self.base_model.predict(np.expand_dims(img, axis=0))
        predicted_label = self.encoded_training_data['knn_classifier'].predict(encoding)
        if with_top5:    
            prediction_top5_idx = self.encoded_training_data['knn_classifier'].kneighbors(encoding, n_neighbors=5)
            prediction_top5 = [self.encoded_training_data['labels'][prediction_top5_idx[1][0][i]] for i in range(5)]
            return predicted_label, prediction_top5
        else:
            return predicted_label

    def calculate_prediction_accuracy(self):
        correct_top1 = 0
        correct_top5 = 0

        accuracies = {'top1':0,
                      'top5':0 }
        total_n_of_images = len(self.data_loader.images_paths['val'])
        for img_path, img_label in zip(self.data_loader.images_paths['val'],
                                       self.data_loader.images_labels['val']):
            prediction, prediction_top5 = self.predict_knn(img_path, with_top5=True)
            if prediction[0] == img_label:
                correct_top1 += 1
            if img_label in prediction_top5:
                correct_top5 += 1
        accuracies['top1'] = correct_top1/total_n_of_images
        accuracies['top5'] = correct_top5/total_n_of_images

        return accuracies

    def test_siamese_oneshot(self, N, k, s="val", verbose=0): #TODO FIXME not sure that this works
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N)
            anchor_path = inputs[0][0]
            anchor_im = cv2.imread(anchor_path)
            anchor_im = cv2.resize(anchor_im, (self.input_shape[1], self.input_shape[0]))
            dist = []
            for j in range(len(inputs[0])):
                test_im = cv2.imread(inputs[1][j])
                test_im = cv2.resize(test_im, (self.input_shape[1], self.input_shape[0]))
                dist.append(self.model.predict([np.array([anchor_im]), np.array([test_im])]))
            if np.argmin(dist) == np.argmax(targets):
                n_correct += 1
            #print(dist, targets)
            #print(np.argmax(targets), np.argmax(dist), np.argmin(dist))
        percent_correct = (100.0 * n_correct / k)
        return percent_correct
    
    
    def test_oneshot(self, N, k, X=None, s="val", verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        if X is None:
            X = prepare_sample_dict(s)
        n_correct = 0
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N, X, s)
            anchor_path = inputs[0][0]
            anchor_im = cv2.imread(anchor_path)
            anchor_im = cv2.resize(anchor_im, (self.input_shape[1], self.input_shape[0]))
            anchor_encoding = self.base_model.predict(np.expand_dims(anchor_im, axis=0))
            dist = []
            for j in range(len(inputs[0])):
                test_im = cv2.imread(inputs[1][j])
                test_im = cv2.resize(test_im, (self.input_shape[1], self.input_shape[0]))
                test_encoding = self.base_model.predict(np.expand_dims(test_im, axis=0))
                dist.append(np.sqrt(np.sum((anchor_encoding - np.array(test_encoding))**2, axis=1)))
            if np.argmin(dist) == np.argmax(targets):
                n_correct += 1
        percent_correct = (100.0 * n_correct / k)
        return percent_correct

    def make_oneshot_task(self, N, X, s="val"):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        n_classes = len(set(self.data_loader.images_labels[s]))
        #X = {}
        #for l in self.data_loader.images_labels[s]:
        #    X[l] = [f for f in self.data_loader.images_paths[s] if s + '/' + l + '/' in f]
        true_category = rng.choice(self.data_loader.images_labels[s], size=(1,), replace=False)[0]
        all_false_cats = [c for c in self.data_loader.images_labels[s] if c != true_category]
        categories = rng.choice(all_false_cats, size=(N,), replace=False)
        ex1, ex2 = rng.choice(len(X[true_category])-1, replace=False, size=(2,))
        if (ex2 >= ex1):
            ex2+=1
        
        test_image = X[true_category][ex1]
        
        support_set = []
        for l in categories:
            idx = rng.randint(0, len(X[l]), size=(1,))
            support_set.append(X[l][idx[0]])
        support_set[0] = X[true_category][ex2]
        
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, [test_image for i in range(N)], support_set)
        pairs = [test_image, support_set]

        return pairs, targets

    def prepare_sample_dict(self, s):
        X = {}
        for l in self.data_loader.images_labels[s]:
            X[l] = [f for f in self.data_loader.images_paths[s] if s + '/' + l + '/' in f]
        return X