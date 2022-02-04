import fiftyone as fo
import fiftyone.zoo as foz

train_dataset = foz.load_zoo_dataset("coco-2014", split="train")
evaluate_dataset = foz.load_zoo_dataset("coco-2014", split="validation")
test_dataset = foz.load_zoo_dataset("coco-2014", split="test")