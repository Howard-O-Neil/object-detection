import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import mylib.io_utils.VOC_2012 as io_VOC_2012
import mylib.img_utils as imu
import logging
import os


class Kaming_he_dense(keras.layers.Layer):
    fan_in = 2  # default, make no change to variables
    units = 1  # draft value
    _lambda = 0.0
    activation = False

    def __init__(self, units, _lambda, activation=True, dropout_rate=0.):
        super(Kaming_he_dense, self).__init__()
        self.units = units
        self._lambda = _lambda
        self.activation = activation
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.fan_in = input_shape[-1]
        he_scale = tf.math.sqrt(tf.math.divide(2.0, self.fan_in))

        w_init = tf.multiply(tf.random.normal([input_shape[-1], self.units]), he_scale)
        b_init = tf.multiply(tf.random.normal([self.units]), he_scale)

        self.w = tf.Variable(w_init, dtype=tf.float32, trainable=True)
        self.b = tf.Variable(b_init, dtype=tf.float32, trainable=True)

    def call(self, inputs, training=True):
        regularized_loss = tf.math.multiply(
            self._lambda, tf.reduce_sum(tf.square(tf.nn.bias_add(self.w, self.b)))
        )
        self.add_loss(regularized_loss)

        if self.activation:
            if training:
                return tf.nn.dropout(
                    tf.nn.tanh(tf.nn.bias_add(tf.matmul(inputs, self.w), self.b)),
                    rate=self.dropout_rate,
                )
            else:
                return tf.nn.tanh(tf.nn.bias_add(tf.matmul(inputs, self.w), self.b))
        else:
            return tf.nn.bias_add(tf.matmul(inputs, self.w), self.b)


class Bbox_predict:
    propose_regions = np.array([])
    ground_truth_bbbox = np.array([])
    imgs_df = np.array([])
    list_imgs = np.array([])
    img_cnn_model = None

    img_batch_size = 1
    train_batch_size = 16
    _lambda = 100.0
    epochs = 50

    def __init__(self, version=1):
        self.model = keras.Sequential(
            [
                Kaming_he_dense(2048, self._lambda, dropout_rate=0.45),
                Kaming_he_dense(2048, self._lambda, dropout_rate=0.45),
                Kaming_he_dense(1024, self._lambda, dropout_rate=0.35),
                Kaming_he_dense(1024, self._lambda, dropout_rate=0.35),
                Kaming_he_dense(1024, self._lambda, dropout_rate=0.35),
                Kaming_he_dense(512, self._lambda, dropout_rate=0.25),
                Kaming_he_dense(512, self._lambda, dropout_rate=0.25),
                Kaming_he_dense(512, self._lambda, dropout_rate=0.25),
                Kaming_he_dense(256, self._lambda, dropout_rate=0.15),
                Kaming_he_dense(256, self._lambda, dropout_rate=0.15),
                Kaming_he_dense(256, self._lambda, dropout_rate=0.15),
                Kaming_he_dense(4, self._lambda, activation=False),
            ]
        )
        self.model.build((None, 4096))

        self.optimizer = keras.optimizers.SGD(learning_rate=0.000001, momentum=0.9)
        self.model_path = f"""{os.getenv("model_path")}/model{version}"""

        self.logger = logging.getLogger("r-cnn logger")
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(os.getenv("log_path"), mode="w")
        stream_handler = logging.StreamHandler()
        
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.addHandler(stream_handler)

    def assign_img_list_train(self, list_imgs):
        self.list_imgs = list_imgs

    def assign_img_list_validation(self, list_imgs):
        self.list_imgs_validation = list_imgs

    def assign_bbox_dataset(self, regions, gts, imgs):
        self.propose_regions = regions
        self.ground_truth_bbbox = gts
        self.imgs_df = imgs

    def assign_cnn_model(self, cnn_model):
        self.img_cnn_model = cnn_model

    def loss_fn(self, train_batch_x, train_batch_y, predictions):
        gt_x = train_batch_y[:, 0]
        gt_y = train_batch_y[:, 1]
        gt_w = train_batch_y[:, 2]
        gt_h = train_batch_y[:, 3]

        propose_x = train_batch_x[:, 0]
        propose_y = train_batch_x[:, 1]
        propose_w = train_batch_x[:, 2]
        propose_h = train_batch_x[:, 3]

        tx = tf.math.divide(tf.math.subtract(gt_x, propose_x), propose_w)
        ty = tf.math.divide(tf.math.subtract(gt_y, propose_y), propose_h)
        tw = tf.math.log(tf.math.divide(gt_w, propose_w))
        th = tf.math.log(tf.math.divide(gt_h, propose_h))

        lx = tf.math.subtract(tx, predictions[:, 0])
        ly = tf.math.subtract(ty, predictions[:, 1])
        lw = tf.math.subtract(tw, predictions[:, 2])
        lh = tf.math.subtract(th, predictions[:, 3])

        loss = tf.math.square(tf.stack([lx, ly, lw, lh], axis=1))
        loss = tf.reduce_mean(tf.reduce_min(loss, axis=1))
        return loss

    @tf.function(experimental_relax_shapes=True)
    # pure Tensor operations, no numpy
    def train_step(self, train_batch_x, train_batch_y, train_batch_img):
        with tf.GradientTape() as tape:
            feature_maps = self.img_cnn_model(train_batch_img)
            bbox_predicts = self.model(feature_maps, training=True)

            loss = self.loss_fn(train_batch_x, train_batch_y, bbox_predicts)

            total_regularized_loss = tf.constant(0.0)
            for reg_loss in self.model.losses:
                total_regularized_loss = tf.add(total_regularized_loss, reg_loss)

            loss = tf.add(loss, total_regularized_loss)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return loss

    @tf.function(experimental_relax_shapes=True)
    # pure Tensor operations, no numpy
    def validate_step(self, train_batch_x, train_batch_y, train_batch_img):
        feature_maps = self.img_cnn_model(train_batch_img)
        bbox_predicts = self.model(feature_maps, training=False)

        loss = self.loss_fn(train_batch_x, train_batch_y, bbox_predicts)

        return loss

    def enumerate_batch(
        self, prefix_str, epoch_id, batch_train, train_batch_size, callback
    ):
        batch_x = np.array([])
        batch_y = np.array([])
        batch_img = np.array([])

        for img in batch_train:
            img = img.decode("utf-8")
            [imgs, _] = io_VOC_2012.scale_imgs([img])

            filter_ids = np.where(self.imgs_df == img, True, False)

            propose = self.propose_regions[filter_ids]
            gt = self.ground_truth_bbbox[filter_ids]
            img_bboxs = imu.extract_bboxs(imgs[0], propose)

            if batch_x.shape[0] <= 0:
                batch_x = propose
                batch_y = gt
                batch_img = img_bboxs
            else:
                batch_x = np.concatenate((batch_x, propose), axis=0)
                batch_y = np.concatenate((batch_y, gt), axis=0)
                batch_img = np.concatenate((batch_img, img_bboxs), axis=0)

        total_train_batch = batch_x.shape[0] // train_batch_size + 1

        if batch_x.shape[0] % train_batch_size == 0:
            total_train_batch -= 1

        for k in range(total_train_batch):
            offset = (k * train_batch_size) % batch_x.shape[0]

            train_batch_x = batch_x[offset : offset + train_batch_size]
            train_batch_y = batch_y[offset : offset + train_batch_size]
            train_batch_img = batch_img[offset : offset + train_batch_size]

            loss = callback(train_batch_x, train_batch_y, train_batch_img).numpy()

            self.logger.info(f"{prefix_str} EPOCH_ID: {epoch_id}, BATCH LOSS: {loss}")

    def save_model(self):
        self.model.save_weights(self.model_path)

    def load_model(self):
        self.model.load_weights(self.model_path)

    def train_loop(self, transfer=False):
        if transfer: self.load_model()
        
        dataset = tf.data.Dataset.from_tensor_slices(self.list_imgs)
        validate_dataset = tf.data.Dataset.from_tensor_slices(self.list_imgs_validation)

        for e in range(self.epochs):

            shuffle_ds = dataset.shuffle(buffer_size=800).batch(self.img_batch_size)
            for _, batch_train in enumerate(shuffle_ds):
                self.enumerate_batch(
                    "[TRAINING]", e, batch_train.numpy(), 16, self.train_step
                )

            validation_batch = validate_dataset.batch(self.img_batch_size)
            for _, batch_train in enumerate(validation_batch):
                self.enumerate_batch(
                    "[VALIDATION]", e, batch_train.numpy(), 16, self.validate_step
                )

        self.save_model()
