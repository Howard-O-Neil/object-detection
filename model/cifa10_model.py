from this import d
import tensorflow_core as tf
import lib.tf_preprocess_img as tf_preproc
import lib.model_hyper_params as mhp
from lib.VGG_16_model import VGG_16
import numpy as np
from datetime import datetime


class cifa10_model:
    def __init__(self, id):
        self.vgg_model = VGG_16()

        self.init_placeholder()

        self.vgg_model.init_conv_parameters()
        self.vgg_model.init_dense_parameters()
        self.init_softmax_parameters()

        self.sess = None
        self.sess_saver = None

        self.train_model = self.get_model(True)
        self.predict_model = self.get_model(False)

        self.EPOCH_id = tf.Variable(0., name="EPOCH_id")
        self.checkpoint_dir = "/content/meta"
        self.MODEL_ID = id

    def get_session(self): return self.sess

    def init_placeholder(self):
        # 32 x 32 images
        self.x = tf.placeholder(tf.float32, [None, 3072])

        # one-hot vector
        self.y = tf.placeholder(tf.float32, [None, 10])

    def init_softmax_parameters(self):
        self.SUM_softmax = tf.constant(0.)
        self.COUNT_softmax = tf.constant(0.)

        fan_in = mhp.num_dense_neurons[1]
        he_norm = tf.cast(tf.sqrt(
            tf.divide(2., np.float32(fan_in))
        ), tf.float32)

        self.SOFTMAX_w = tf.Variable(tf.multiply(
            tf.random_normal([fan_in, 10]),
            he_norm
        ), name="SOFTMAX_w")

        self.SOFTMAX_b = tf.Variable(tf.multiply(
            tf.random_normal([10]),
            he_norm
        ), name="SOFTMAX_b")

        self.SUM_softmax = tf.add(
            self.SUM_softmax,
            tf.reduce_sum(tf.square(
                tf.nn.bias_add(self.SOFTMAX_w, self.SOFTMAX_b))   
            )
        )
        self.COUNT_softmax = tf.add(
            self.COUNT_softmax, tf.cast(fan_in * 10, dtype=tf.float32)
        )
    
    def get_sum_parameters(self):
        return tf.add(
            self.vgg_model.get_sum_params(),
            tf.reduce_sum(
                tf.square(tf.nn.bias_add(self.SOFTMAX_w, self.SOFTMAX_b))
            )
        )

    def pre_process_input(self, is_train):
        preprocess_x = tf_preproc.tf_preprocess_images(self.x, is_train)

        x_scale = None
        self.batch_x = self.x
        self.batch_y = self.y

        if is_train:
            x_scale = tf_preproc.tf_augment_images(preprocess_x)
            x_data_shape = tf.shape(x_scale)
            x_data_size = tf.squeeze(tf.slice(x_data_shape, [0], [1]))

            y_data_shape = tf.shape(self.y)
            y_data_size = tf.squeeze(tf.slice(y_data_shape, [0], [1]))

            y_augments = tf.tile(self.y, [tf.cast(tf.divide(x_data_size, y_data_size), tf.int32), 1]) 

            shuffle_index = tf.random.shuffle(
                tf.range(0, x_data_size, 1)
            )

            self.batch_x = tf.gather(x_scale, shuffle_index)
            self.batch_y = tf.gather(y_augments, shuffle_index)
        else:
            self.batch_x = preprocess_x
        
    def get_model(self, is_train):
        self.pre_process_input(is_train)

        vgg =  self.vgg_model.get_model(self.batch_x, is_train)

        return tf.nn.bias_add(tf.matmul(vgg, self.SOFTMAX_w), self.SOFTMAX_b)
    
    def get_cost_train(self):
        return tf.add(
            tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.batch_y, logits=self.train_model)),
            tf.multiply(
                mhp.lambda_val,
                self.get_sum_parameters()
            )
        )

    def get_cost_predict(self):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.batch_y, logits=self.predict_model))

    def compare_dif_result(self, x1, x2, is_count):
        total_same_val = 0
        for i, x in enumerate(x1):
            if x1[i].astype(np.int32) == \
                x2[i].astype(np.int32):
                total_same_val += 1
        
        if is_count: return total_same_val
        return total_same_val / x1.shape[0]

    def cost_by_batch(self, x_dataset, y_dataset, bs):
        ttb = y_dataset.shape[0] // bs

        avg_res = 0.0
        for i in range(0, ttb):
            offset = (i * bs) % y_dataset.shape[0]
            bx = x_dataset[offset:offset+bs, :]
            by = y_dataset[offset:offset+bs, :]

            avg_res += self.sess.run(self.get_cost_predict(), feed_dict={self.x: bx, self.y: by})
        
        return avg_res / ttb
    
    def precision_by_batch(self, x_dataset, y_dataset, bs):
        predict = tf.argmax(self.predict_model, axis=1)

        ttb = y_dataset.shape[0] // bs

        avg_res = 0
        for i in range(0, ttb):
            offset = (i * bs) % y_dataset.shape[0]
            bx = x_dataset[offset:offset+bs, :]
            by = y_dataset[offset:offset+bs]

            avg_res += self.compare_dif_result(
                self.sess.run(predict, feed_dict={self.x: bx}), by, True
            )
        
        return avg_res / y_dataset.shape[0]


    def init_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.sess_saver = tf.train.Saver()

    def save_model(self):
        self.sess_saver.save(self.sess, \
            f"{self.checkpoint_dir}/model{self.MODEL_ID}/obj-detect-{self.MODEL_ID}")
    
    def training_loop(self, data, y_train_one_hot, t_data_eva, t_labels_eva, y_test_one_hot_eva):
        # PLEASE DONT RUN THIS
        # MODEL HAS BEEN TRAINED, RESTORE TO RUN THE DEMO

        train_op = tf.train.AdamOptimizer(learning_rate=mhp.learning_rate) \
            .minimize(self.get_cost_train())

        total_batch = (data.shape[0] // mhp.batch_size) + 1

        print('TOTAL BATCH: ', total_batch)
        print(mhp.training_epochs)

        print(f"===== [START TRAINING] {str(datetime.now())} =====")
        print()

        stable_cost_variance_rate = 0.000001

        avg_train_cost = []
        avg_evaluate_cost = []
        avg_evaluate_precision = []

        for j in range(0, mhp.training_epochs):
            avg_train_b = []
            for i in range(0, total_batch):
                offset = (i * mhp.batch_size) % labels.shape[0]

                batch_data = data[offset:offset+mhp.batch_size, :]
                batch_onehot_vals = y_train_one_hot[offset:offset+mhp.batch_size, :]

                _, cost_val = self.sess.run([train_op, self.get_cost_train()], \
                    feed_dict={self.x: batch_data, self.y: batch_onehot_vals})        

                print(cost_val)

                avg_train_b.append(cost_val)
            
            print(f"=== [START EVALUATE] {str(datetime.now())} ===")
            avg_train = np.mean(np.array(avg_train_b))
            avg_evaluate = self.cost_by_batch(t_data_eva, y_test_one_hot_eva, 128)
            avg_evaluate_preci = self.precision_by_batch(t_data_eva, t_labels_eva, 128)

            avg_train_cost.append(avg_train)
            avg_evaluate_cost.append(avg_evaluate)
            avg_evaluate_precision.append(avg_evaluate_preci)

            print('Epoch {}. AVG train cost {}'.format(j, avg_train))
            print('Epoch {}. Evaluation cost {}'.format(j, avg_evaluate))
            print('Epoch {}. Evaluation precision {}'.format(j, avg_evaluate_preci))
            
            if len(avg_evaluate_cost) > 3:
                if abs(avg_evaluate_cost[-1] - avg_evaluate_cost[-2]) <= stable_cost_variance_rate and \
                    abs(avg_evaluate_cost[-2] - avg_evaluate_cost[-3]) <= stable_cost_variance_rate:

                    self.save_model()
                    
                    break

            print(f"=== [END EVALUATE] {str(datetime.now())} ===")

            self.sess.run(tf.assign(self.EPOCH_id, tf.add(self.EPOCH_id, 1.)))

            # shuffle all the dataset
            arr_idx_t = np.arange(data.shape[0])
            np.random.shuffle(arr_idx_t)

            data = data[arr_idx_t]
            labels = labels[arr_idx_t]

            y_train_one_hot = y_train_one_hot[arr_idx_t]

        print(f"===== [END TRAINING] {str(datetime.now())} =====")
        # sess.close()