# Filter bbox with label contain "car"
boxes_label = lb.labels[predictions[:, 0].astype(np.int32)]
label_filter_idx = np.where(
    np.char.find(boxes_label[:, 1], "car")
    or np.char.find(boxes_label[:, 1], "vehicle") != -1,
    True,
    False,
)

print("===== FILTER LABEL WITH CAR =====")
print(boxes_label)
print(label_filter_idx)
print("=====")

car_bboxes = boxes[label_filter_idx]
car_predictions = predictions[label_filter_idx, 1]
nms = tf.image.non_max_suppression_with_scores(
    bbox_xyxy(car_bboxes),
    car_predictions,
    max_output_size=20,
    iou_threshold=0.5,
    score_threshold=0.6,
    soft_nms_sigma=0.5,
)

filter_bboxes = car_bboxes[nms[0].numpy()]

fig, ax = plt.subplots(1)

plt.imshow(img_tensor / 255.0)
for i, rect in enumerate(filter_bboxes):
    r = mpatches.Rectangle(
        (rect[0], rect[1]),
        rect[2],
        rect[3],
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(r)

plt.show()
plt.close()
