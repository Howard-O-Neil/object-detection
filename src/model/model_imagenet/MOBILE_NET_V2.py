import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import mylib.img_utils as imu
import PIL

class Pretrain_mobilenet_v2:
    def __init__(self):
        self.model = keras.applications.MobileNetV2(
            alpha=1.0,
            include_top=True,
            weights="imagenet",
            input_shape=None,
            input_tensor=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )

    def get_model(self):
        return self.model

    def predict_labels(self, imgs):
        if len(imgs.shape) <= 3:
            imgs = np.expand_dims(imgs, 0)

        imgs = keras.applications.mobilenet_v2.preprocess_input(imgs)
        
        computed_features = self.model.predict_on_batch(imgs)

        compute_labels = np.argmax(computed_features, axis=1).astype(np.float32)
        compute_scores = np.max(computed_features, axis=1).astype(np.float32)

        compute_res = np.stack((compute_labels, compute_scores), axis=1)

        return compute_res

    def predict_bboxs(self, original_img, bboxs):
        if len(bboxs.shape) <= 1:
            bboxs = np.expand_dims(bboxs, axis=0)

        batch_size = 16
        total_batch = bboxs.shape[0] // batch_size + 1

        if bboxs.shape[0] % batch_size == 0:
            total_batch -= 1
        
        predictions = np.array([])
        for i in range(total_batch):
            offset = (i * batch_size) % bboxs.shape[0]
            
            sub_bboxs = bboxs[offset:offset+batch_size]
            img_bboxs = keras.applications.mobilenet_v2.preprocess_input(
                imu.extract_bboxs(original_img, sub_bboxs)
            )

            computed_features = self.model.predict_on_batch(img_bboxs)

            compute_labels = np.argmax(computed_features, axis=1).astype(np.float32)
            compute_scores = np.max(computed_features, axis=1).astype(np.float32)

            compute_res = np.stack((compute_labels, compute_scores), axis=1)

            if predictions.shape[0] <= 0:
                predictions = compute_res
            else: predictions = np.concatenate((predictions, compute_res), axis=0)

        return predictions