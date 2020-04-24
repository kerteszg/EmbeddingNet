from keras.layers import Dense, Input, Lambda, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, concatenate
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K


def get_backbone(input_shape,
                 encodings_len=4096,
                 backbone_type='simple',
                 embeddings_normalization=True,
                 backbone_weights='imagenet',
                 freeze_backbone=False):
    if backbone_name == 'simple':
        input_image = Input(input_shape)
        x = Conv2D(64, (10, 10), activation='relu',
                   kernel_regularizer=l2(2e-4))(input_image)
        x = MaxPool2D()(x)
        x = Conv2D(128, (7, 7), activation='relu',
                   kernel_regularizer=l2(2e-4))(x)
        x = MaxPool2D()(x)
        x = Conv2D(128, (4, 4), activation='relu',
                   kernel_regularizer=l2(2e-4))(x)
        x = MaxPool2D()(x)
        x = Conv2D(256, (4, 4), activation='relu',
                   kernel_regularizer=l2(2e-4))(x)
        x = Flatten()(x)
        backbone_model = Model(
            inputs=[input_image], outputs=[x])
        encoded_output = Dense(encodings_len, activation='relu',
                               kernel_regularizer=l2(1e-3))(x)
        if embeddings_normalization:
            encoded_output = Lambda(lambda x: K.l2_normalize(
                x, axis=1), name='l2_norm')(encoded_output)
        base_model = Model(
            inputs=[input_image], outputs=[encoded_output])
    elif backbone_type == 'nist_mymodel':
        input_image = Input(input_shape)
        x = Conv2D(64, (3, 3), activation='relu')(input_image)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(256, activation="relu")(x)
        #x = Dropout(0.5)(x)
        backbone_model = Model(
            inputs=[input_image], outputs=[x])
        encoded_output = Dense(encodings_len, activation='relu')(x)
        base_model = Model(inputs=[input_image], outputs=[encoded_output])
    elif backbone_type == 'nist_sq_mymodel':
        input_image = Input(input_shape)
        x = Conv2D(64, (5, 5), activation='relu')(input_image)
        x = MaxPool2D()(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(256, activation="relu")(x)
        #x = Dropout(0.5)(x)
        backbone_model = Model(
            inputs=[input_image], outputs=[x])
        encoded_output = Dense(encodings_len, activation='relu')(x)
        base_model = Model(inputs=[input_image], outputs=[encoded_output])
    elif backbone_type == 'nist_mdipfl50_mymodel':
        input_image = Input(input_shape)
        x = Conv2D(64, (3, 3), activation='relu')(input_image)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = Conv2D(512, (3, 3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(256, activation="relu")(x)
        #x = Dropout(0.5)(x)
        backbone_model = Model(
            inputs=[input_image], outputs=[x])
        encoded_output = Dense(encodings_len, activation='relu')(x)
        base_model = Model(inputs=[input_image], outputs=[encoded_output])
    else:
        from classification_models.keras import Classifiers
        classifier, preprocess_input = Classifiers.get(backbone_type)
        backbone_model = classifier(input_shape=input_shape, weights=None)
                                    #weights=backbone_weights,
                                    #include_top=False)
        if freeze_backbone:
            for layer in backbone_model.layers[:-2]:
                layer.trainable = False
        
        #after_backbone = backbone_model.output
        x = backbone_model.output
        #x = Flatten()(after_backbone)

        #x = Dense(encodings_len//2, activation="relu")(x)
        x = Dense(1000, activation="relu")(x)

        encoded_output = Dense(encodings_len, activation="relu")(x)
        if embeddings_normalization:
            encoded_output = Lambda(lambda x: K.l2_normalize(
                x, axis=1), name='l2_norm')(encoded_output)
        base_model = Model(
            inputs=[backbone_model.input], outputs=[encoded_output])

    return base_model, backbone_model
