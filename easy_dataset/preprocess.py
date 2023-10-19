# https://www.kaggle.com/code/mbmmurad/prepare-dataset-for-wav2vec2
import re

import pandas as pd
from bnunicodenormalizer import Normalizer
from datasets import Audio
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor

bnorm = Normalizer()

df = pd.read_csv("../../kaggle/input/valid.csv")
val = df[df.split == "valid"]
print("Validation set shape : ", val.shape)

train, val = train_test_split(val, test_size=0.2, random_state=42)

print(train.shape)
print(val.shape)

audio_dir = "../../kaggle/input/train_mp3s/"
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\—\‘\'\‚\“\”\…]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]) + " "
    return batch


def normalize(batch):
    _words = [bnorm(word)['normalized'] for word in batch["sentence"].split()]
    batch["sentence"] = " ".join([word for word in _words if word is not None])
    return batch

#processor = Wav2Vec2Processor.from_pretrained("arijitx/wav2vec2-xls-r-300m-bengali")
processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")


def prepare_dataset(batch):
    #batch["audio"]["array"] = np.trim_zeros(batch["audio"]["array"], 'fb')
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch


def create_dataset(df):
    paths = df['id'].apply(lambda x: audio_dir + x + ".mp3")
    dataset = Dataset.from_dict({"audio": paths, "sentence": df['sentence'].tolist()}).cast_column("audio", Audio(
        sampling_rate=16000))
    dataset = dataset.map(remove_special_characters)
    dataset = dataset.map(normalize)
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

    return dataset


train_ds = create_dataset(train)
train_ds.save_to_disk("train")

val_ds = create_dataset(val)
val_ds.save_to_disk("val")