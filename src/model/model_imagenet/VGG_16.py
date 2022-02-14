import tensorflow as tf
import numpy as np
import mylib.img_utils as imu
import PIL

class Pretrain_VGG16:
    def __init__(self):
        self.VGG_16_model = tf.keras.applications.VGG16(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )
    
    def get_model(self):
        return self.VGG_16_model

    def predict_scores(self, scale_img, bboxs):
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
            img_bboxs = imu.extract_bboxs(scale_img, sub_bboxs)

            computed_features = self.VGG_16_model.predict_on_batch(img_bboxs)

            compute_labels = np.argmax(compute_labels, axis=1).astype(np.float32)
            compute_scores = np.max(computed_features, axis=1).astype(np.float32)

            compute_res = np.stack((compute_labels, compute_scores), axis=1)

            if predictions.shape[0] <= 0:
                predictions = compute_res
            else: predictions = np.concatenate((predictions, compute_res), axis=0)

        return predictions