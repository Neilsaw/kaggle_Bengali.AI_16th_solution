# https://www.kaggle.com/code/mbmmurad/lb-0-49-wav2vec2-baseline-train-and-infer

import datetime
import random
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
from datasets import load_metric, load_from_disk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer

model_path = "./ai4bharat_white_noise_2023-09-05_11-45-56/20epoch"

processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)
model = model.to("cuda")

"""
processor = Wav2Vec2Processor.from_pretrained("arijitx/wav2vec2-xls-r-300m-bengali")
model = Wav2Vec2ForCTC.from_pretrained("./demo_train/checkpoint-13500")
model = model.to("cuda")
"""

def augmentation(batch):
    new_input_values = []
    for input_value in batch["input_values"]:
        np_audio = np.array(input_value)

        # add white noise
        if random.random() > 0.5:
            noise_size = random.uniform(0.025, 0.1)
            wn = np.random.randn(len(np_audio))
            np_audio = np_audio + noise_size * wn

        new_input_values.append(np_audio.tolist())

    aug_batch = {"input_values": new_input_values, "labels": batch["labels"]}
    return aug_batch


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load_metric("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


model.config.ctc_zero_infinity = True

model.freeze_feature_encoder()

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

repo_name = f"wn_high_wer_{formatted_time}"


training_args = TrainingArguments(
    report_to="neptune",
    output_dir=repo_name,
    group_by_length=True,
    #per_device_train_batch_size=4,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    num_train_epochs=30,
    fp16=True,
    gradient_checkpointing=True,
    save_strategy="epoch",
    logging_strategy="epoch",
    #learning_rate=5e-5,
    learning_rate=5e-5,
    weight_decay=0.05,
    warmup_steps=500,
    save_total_limit=5,
)

train_dataset = load_from_disk("../../kaggle/input/preprocessed-train--wer-0_3_1_5--cer-0_15--mos-2/train")
val_dataset = load_from_disk("../../kaggle/input/preprocessed-train--wer-0_3_1_5--cer-0_15--mos-2/val")

#train_dataset = load_from_disk("../../kaggle/input/test/train")
#val_dataset = load_from_disk("../../kaggle/input/test/val")

# with augmentation
train_dataset.set_transform(augmentation)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.feature_extractor,
)


trainer.train()


