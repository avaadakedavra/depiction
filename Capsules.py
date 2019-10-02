import os
import sys
import numpy as np
from random import shuffle
import tensorflow as tf
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, concatenate, Multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.utils import plot_model
import shutil
from PIL import Image
from encode import *

class Capsules(object):
    model_name = 'capsules'

    def __init__(self):
        K.set_image_dim_ordering('tf')
        self.generator = None
        self.discriminator = None
        self.model = None
        self.img_width = 64
        self.img_height = 64
        self.img_channels = 3
        self.random_input_dim = 100
        self.text_input_dim = 300 #100 (256)
        self.config = None
        self.glove_source_dir_path = './very_large_data'
        self.glove_model = GloveModel()

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, Capsules.model_name + '-config.npy')

    @staticmethod
    def get_weight_file_path(model_dir_path, model_type):
        return os.path.join(model_dir_path, Capsules.model_name + '-' + model_type + '-weights.h5')

    def create_model(self):
        init_img_width = self.img_width // 4
        init_img_height = self.img_height // 4

        #GENERATOR ARCHITECTURE
        random_input = Input(shape=(self.random_input_dim,))
        text_input1 = Input(shape=(self.text_input_dim,))
        random_dense = Dense(self.random_input_dim)(random_input)
        text_layer1 = Dense(1024)(text_input1)
        
        merged = concatenate([random_dense, text_layer1])
        
        generator_layer = Dense(128 * init_img_width * init_img_height, activation='relu')(merged)
        generator_layer = BatchNormalization(momentum=0.9)(generator_layer)
        generator_layer = LeakyReLU(alpha=0.1)(generator_layer)
        generator_layer = Reshape((init_img_width, init_img_height , 128)) (generator_layer)
        
        generator_layer = Conv2D(128,kernel_size=4 , strides=1 , padding='same')(generator_layer)
        generator_layer = BatchNormalization(momentum=0.9)(generator_layer)
        generator_layer = LeakyReLU(alpha=0.1)(generator_layer)
        
        generator_layer = Conv2DTranspose(128,kernel_size=4 , strides=2 , padding='same')(generator_layer)
        generator_layer = BatchNormalization(momentum=0.9)(generator_layer)
        generator_layer = LeakyReLU(alpha=0.1)(generator_layer)
        
        generator_layer = Conv2D(128,kernel_size=5 , strides=1 , padding='same')(generator_layer)
        generator_layer = BatchNormalization(momentum=0.9)(generator_layer)
        generator_layer = LeakyReLU(alpha=0.1)(generator_layer)
        
        generator_layer = Conv2DTranspose(128,kernel_size=4 , strides=2 , padding='same')(generator_layer)
        generator_layer = BatchNormalization(momentum=0.9)(generator_layer)
        generator_layer = LeakyReLU(alpha=0.1)(generator_layer)
        
        generator_layer = Conv2D(128, kernel_size=5 , strides=1 , padding='same')(generator_layer)
        generator_layer = BatchNormalization(momentum=0.9)(generator_layer)
        generator_layer = LeakyReLU(alpha=0.1)(generator_layer)
        
        generator_layer = Conv2D(128, kernel_size=5 , strides=1 , padding='same')(generator_layer)
        generator_layer = BatchNormalization(momentum=0.9)(generator_layer)
        generator_layer = LeakyReLU(alpha=0.1)(generator_layer)
        
        generator_layer = Conv2D(3,kernel_size= 5 , strides=1 , padding='same')(generator_layer)
        generator_output = Activation('tanh')(generator_layer)
        
        self.generator = Model([random_input, text_input1], generator_output)
        
        print('\nGENERATOR:\n')
        print('generator: ', self.generator.summary())
        print("\n\n")
        
        self.generator.compile(loss='binary_crossentropy', optimizer= Adam(0.0002,0.5), metrics=["accuracy"])
        #plot_model(self.generator, to_file = "generator_model.jpg", show_shapes = True)
        
        
        #DISCRIMINATOR MODEL        
        text_input2 = Input(shape=(self.text_input_dim,))
        text_layer2 = Dense(1024)(text_input2)

        img_input2 = Input(shape=(self.img_width, self.img_height, self.img_channels))
        
        # first typical convlayer outputs a 20x20x256 matrix
        img_layer2 = Conv2D(filters=256, kernel_size=8 , strides=1, padding='valid', name='conv1')(img_input2)  #kernel_size=9
        img_layer2 = LeakyReLU()(img_layer2)
        
        # original 'Dynamic Routing Between Capsules' paper does not include the batch norm layer after the first conv group
        img_layer2 = BatchNormalization(momentum=0.8)(img_layer2)
        
        img_layer2 = Conv2D(filters=256, kernel_size=8 , strides=2, padding='valid', name='conv2')(img_layer2)  #kernel_size=9
        img_layer2 = LeakyReLU()(img_layer2)
        
        # original 'Dynamic Routing Between Capsules' paper does not include the batch norm layer after the first conv group
        img_layer2 = BatchNormalization(momentum=0.8)(img_layer2)
        
        #NOTE: Capsule architecture starts from here.
        # primarycaps coming first
    
        # filters 512 (n_vectors=8 * channels=64)
        img_layer2 = Conv2D(filters=8 * 64, kernel_size=8, strides=2, padding='valid', name='primarycap_conv2_1')(img_layer2)
    
        #img_layer2 = Conv2D(filters=8 * 64, kernel_size=8, strides=2, padding='valid', name='primarycap_conv2_2')(img_layer2)
        
        # reshape into the 8D vector for all 32 feature maps combined
        # (primary capsule has collections of activations which denote orientation of the digit
        # while intensity of the vector which denotes the presence of the digit)
        img_layer2 = Reshape(target_shape=[-1, 8], name='primarycap_reshape')(img_layer2)
    
        # the purpose is to output a number between 0 and 1 for each capsule where the length of the input decides the amount
        img_layer2 = Lambda(squash, name='primarycap_squash')(img_layer2)
        img_layer2 = BatchNormalization(momentum=0.8)(img_layer2)

        # digitcaps are here        
        img_layer2 = Flatten()(img_layer2)
        
        # capsule (i) in a lower-level layer needs to decide how to send its output vector to higher-level capsules (j)
        # it makes this decision by changing scalar weight (c=coupling coefficient) that will multiply its output vector and then be treated as input to a higher-level capsule
        # uhat = prediction vector, w = weight matrix but will act as a dense layer, u = output from a previous layer
        # uhat = u * w
        # neurons 160 (num_capsules=102 * num_vectors=16)
    
        uhat = Dense(1632, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')(img_layer2)
    
        # c = coupling coefficient (softmax over the bias weights, log prior) | "the coupling coefficients between capsule (i) and all the capsules in the layer above sum to 1"
        # we treat the coupling coefficiant as a softmax over bias weights from the previous dense layer
        c = Activation('softmax', name='softmax_digitcaps1')(uhat) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
    
        # s_j (output of the current capsule level) = uhat * c
        c = Dense(1632)(c) # compute s_j
        x = Multiply()([uhat, c])
        s_j = LeakyReLU()(x)

        # we will repeat the routing part 2 more times (num_routing=3) to unfold the loop
        c = Activation('softmax', name='softmax_digitcaps2')(s_j) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
        c = Dense(1632)(c) # compute s_j
        x = Multiply()([uhat, c])
        s_j = LeakyReLU()(x)

        c = Activation('softmax', name='softmax_digitcaps3')(s_j) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
        c = Dense(1632)(c) # compute s_j
        x = Multiply()([uhat, c])
        s_j = LeakyReLU()(x)
        
        merged = concatenate([s_j, text_layer2])
        
        discriminator_layer = Activation('relu')(merged)
        pred = Dense(1, activation='sigmoid')(discriminator_layer)
        
        self.discriminator = Model([img_input2, text_input2], pred)
             
        print('\nDISCRIMINATOR:\n')
        print('discriminator: ', self.discriminator.summary())
        print("\n\n")
        
        self.discriminator.compile(loss='binary_crossentropy', optimizer= Adam(0.0002, 0.5), metrics=['accuracy'])
        #plot_model(self.discriminator, to_file = "discriminator_model.jpg", show_shapes = True)
        
        
        #ADVERSARIAL MODEL
        model_output = self.discriminator([self.generator.output, text_input1])
        self.model = Model([random_input, text_input1], model_output)
        self.discriminator.trainable = False
        
        print('\nADVERSARIAL MODEL:\n')
        print('generator-discriminator:\n', self.model.summary())
        print("\n\n")
        
        self.model.compile(loss='binary_crossentropy', optimizer= Adam(0.0002, 0.5) , metrics=["accuracy"])
        #plot_model(self.model, to_file = "model.jpg", show_shapes = True)
        
        #self.model.save("capgans.h5")
        #json_string = self.model.to_json()

    def load_model(self, model_dir_path):
        
        config_file_path = Capsules.get_config_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.img_width = self.config['img_width']
        self.img_height = self.config['img_height']
        self.img_channels = self.config['img_channels']
        self.random_input_dim = self.config['random_input_dim']
        self.text_input_dim = self.config['text_input_dim']
        self.glove_source_dir_path = self.config['glove_source_dir_path']
        self.create_model()
        self.glove_model.load(self.glove_source_dir_path, embedding_dim=self.text_input_dim)
        self.generator.load_weights(Capsules.get_weight_file_path(model_dir_path, 'generator'))
        self.discriminator.load_weights(Capsules.get_weight_file_path(model_dir_path, 'discriminator'))

    def fit(self, model_dir_path, image_label_pairs, epochs=None, batch_size=None, snapshot_dir_path=None,
            snapshot_interval=None):
        # Change the epoch value
        if epochs is None:
            epochs = 100

        if batch_size is None:
            batch_size = 64

        if snapshot_interval is None:
            snapshot_interval = 20
        
        self.config = dict()
        self.config['img_width'] = self.img_width
        self.config['img_height'] = self.img_height
        self.config['random_input_dim'] = self.random_input_dim
        self.config['text_input_dim'] = self.text_input_dim
        self.config['img_channels'] = self.img_channels
        self.config['glove_source_dir_path'] = self.glove_source_dir_path

        self.glove_model.load(data_dir_path=self.glove_source_dir_path, embedding_dim=self.text_input_dim)

        config_file_path = Capsules.get_config_file_path(model_dir_path)

        np.save(config_file_path, self.config)
        noise = np.zeros((batch_size, self.random_input_dim))
        text_batch = np.zeros((batch_size, self.text_input_dim))

        self.create_model()

        batch_count = int(image_label_pairs.shape[0] / batch_size)
        print(batch_count)
        
        #exp_replay = [] # array to store sample for experience replay
        
        for epoch in range(epochs):
          cum_d_loss = 0
          cum_g_loss = 0
          cum_g_acc = 0
          cum_d_acc = 0
          
          print ('-'*15, 'Epoch %d' % epoch, '-'*15)
          for batch_index in tqdm(range(batch_count)):  
            
            # Step 1: train the discriminator
            image_label_pair_batch = image_label_pairs[batch_index * batch_size:(batch_index + 1) * batch_size]
            
            image_batch = []
            for index in range(batch_size):
              image_label_pair = image_label_pair_batch[index]
              normalized_img = image_label_pair[0]
              text = image_label_pair[1]
              image_batch.append(normalized_img)
              text_batch[index, :] = self.glove_model.encode_doc(text, self.text_input_dim)
              noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)

            image_batch = np.array(image_batch)
            #image_batch = np.transpose(image_batch, (0, 2, 3, 1))
            generated_images = self.generator.predict([noise, text_batch], verbose=0)
            
            # Train on soft targets (add noise to targets as well)
            noise_prop = 0.05 # Randomly flip 5% of targets
            
            # Prepare labels for real data
            true_labels = np.zeros((batch_size, 1)) + np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
            flipped_idx = np.random.choice(np.arange(len(true_labels)), size=int(noise_prop*len(true_labels)))
            true_labels[flipped_idx] = 1 - true_labels[flipped_idx]
            
            # Prepare labels for generated data
            gene_labels = np.ones((batch_size, 1)) - np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
            flipped_idx = np.random.choice(np.arange(len(gene_labels)), size=int(noise_prop*len(gene_labels)))
            gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]
            
            '''
            # Store a random point for experience replay
            r_idx = np.random.randint(batch_size)
            exp_replay.append([generated_images[r_idx], text_batch[r_idx], gene_labels[r_idx]])
            '''
    
            if (epoch * batch_size + batch_index) % snapshot_interval == 0 and snapshot_dir_path is not None:
              self.save_snapshots(generated_images, snapshot_dir_path=snapshot_dir_path,
                                  epoch=epoch, batch_index=batch_index)

            self.discriminator.trainable = True
              
            # Train discriminator on real data
            d_loss_true = self.discriminator.train_on_batch([image_batch, text_batch], true_labels)
              
            # Train discriminator on generated data
            d_loss_gene = self.discriminator.train_on_batch([generated_images, text_batch], gene_labels)
                            
            d_loss = ((np.asarray(d_loss_true) + np.asarray(d_loss_gene))*0.5).tolist()
              
            cum_d_loss += d_loss[0]
            cum_d_acc += d_loss[1]
              
            '''
            #Adversarial Model Training
            #If we have enough points, do experience replay
            if len(exp_replay) == batch_size:
              generated_images = np.array([p[0] for p in exp_replay])
              text_batch = np.array([p[1] for p in exp_replay])
              gene_labels = np.array([p[2] for p in exp_replay])
              expprep_loss_gene = self.discriminator.train_on_batch([generated_images, text_batch], gene_labels)
              exp_replay = []
              break
            ''' 
            
            #step 2: train the generator
            for index in range(batch_size):
              image_label_pair = image_label_pair_batch[index]
              text = image_label_pair[1]
              text_batch[index, :] = self.glove_model.encode_doc(text, self.text_input_dim)
              noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
            
            self.discriminator.trainable = False
            g_loss = self.model.train_on_batch([noise, text_batch], np.zeros((batch_size, 1)))
            
            cum_g_loss += g_loss[0]
            cum_g_acc += g_loss[1]
             
            #for index in range(batch_size):
              #noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
              #g_loss = self.model.train_on_batch([noise, text_batch], np.array([1] * batch_size))
                
          #print('\tEpoch: {}, Generator Loss: {}, Discriminator Loss: {}'.format(epoch+1, cum_g_loss/batch_count, cum_d_loss/batch_count))
          print('\tEpoch: {}, Generator Loss: {}, Generator Accuracy: {}, Discriminator Loss: {}, Disciminator Accuracy: {}'.format(epoch+1, cum_g_loss/batch_count, cum_g_acc/batch_count, cum_d_loss/batch_count, cum_d_acc/batch_count))
          D_L.append(cum_d_loss/batch_count)
          D_A.append(cum_d_acc/batch_count)
          G_L.append(cum_g_loss/batch_count)
          G_A.append(cum_g_acc/batch_count)
              
          #if (epoch * batch_size + batch_index) % 10 == 9:
          self.generator.save_weights(Capsules.get_weight_file_path(model_dir_path, 'generator'), True)
          self.discriminator.save_weights(Capsules.get_weight_file_path(model_dir_path, 'discriminator'), True)
              
        self.generator.save_weights(Capsules.get_weight_file_path(model_dir_path, 'generator'), True)
        self.discriminator.save_weights(Capsules.get_weight_file_path(model_dir_path, 'discriminator'), True)
            
        callbacks.History()
        callbacks.ModelCheckpoint(os.path.join(model_dir_path,'capgans.h5'), monitor='cum_d_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    def generate_image_from_text(self, text):
        noise = np.zeros(shape=(1, self.random_input_dim))
        encoded_text = np.zeros(shape=(1, self.text_input_dim))
        encoded_text[0, :] = self.glove_model.encode_doc(text)
        noise[0, :] = np.random.uniform(-1, 1, self.random_input_dim)
        generated_images = self.generator.predict([noise, encoded_text], verbose=0)
        generated_image = generated_images[0]
        generated_image = generated_image * 127.5 + 127.5
        return Image.fromarray(generated_image.astype(np.uint8))


    def save_snapshots(self, generated_images, snapshot_dir_path, epoch, batch_index):
        image = combine_normalized_images(generated_images)
        img_from_normalized_img(image).save(
            os.path.join(snapshot_dir_path, Capsules.model_name + '-' + str(epoch) + "-" + str(batch_index) + ".jpg"))
