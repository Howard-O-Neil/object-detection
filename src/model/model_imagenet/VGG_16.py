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

    def predict_scores(self, img_dir, bboxs):
        if len(bboxs.shape) <= 1:
            bboxs = np.expand_dims(bboxs, axis=0)

        batch_size = 16
        total_batch = bboxs.shape[0] // batch_size + 1

        # img_dir = img_dir.decode("utf-8")
        img_tensor = tf.cast(
            tf.convert_to_tensor(np.asarray(PIL.Image.open(img_dir))),
            tf.dtypes.float32,
        )
        scale_img = tf.image.resize(
            img_tensor, [500, 500], method="bilinear", preserve_aspect_ratio=True
        ).numpy()

        if bboxs.shape[0] % batch_size == 0:
            total_batch -= 1
        
        predictions = np.array([])
        for i in range(total_batch):
            offset = (i * batch_size) % bboxs.shape[0]
            
            sub_bboxs = bboxs[offset:offset+batch_size]
            img_bboxs = imu.extract_bboxs(scale_img, sub_bboxs)

            compute_labels = np.max(self.VGG_16_model.predict_on_batch(img_bboxs), axis=1) 

            if predictions.shape[0] <= 0:
                predictions = compute_labels
            else: predictions = np.concatenate((predictions, compute_labels), axis=0)

        return predictions