import fiftyone as fo
import fiftyone.zoo as foz

train_dataset = foz.load_zoo_dataset("voc-2012", split="train")
evaluate_dataset = foz.load_zoo_dataset("voc-2012", split="validation")
# test_dataset = foz.load_zoo_dataset("voc-2012", split="test")