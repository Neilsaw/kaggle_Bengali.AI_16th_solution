# https://www.kaggle.com/code/mbmmurad/prepare-dataset-for-wav2vec2
import re

import numpy as np
import pandas as pd
from bnunicodenormalizer import Normalizer
from datasets import Audio
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import Wav2Vec2Processor

bnorm = Normalizer()

df = pd.read_csv("../../kaggle/input/wer-0_3_1_5---cer-0_15---mos-2_data.csv")

df = df.groupby(['sentence', 'client_idx']).head(1).groupby('sentence').head(1)

df = df.reset_index()

df['ykg_wer'] = (df['ykg_wer'] * 10).round()

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df['ykg_wer']), 1):
    df.loc[val_idx, 'fold'] = fold

df['fold'] = df['fold'].astype(np.uint8)
print(df.groupby('fold').size())

audio_dir = "../../kaggle/input/train_mp3s/"
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\—\‘\'\‚\“\”\…]'


def create():
    def remove_special_characters(batch):
        batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]) + " "
        return batch

    def normalize(batch):
        _words = [bnorm(word)['normalized'] for word in batch["sentence"].split()]
        batch["sentence"] = " ".join([word for word in _words if word is not None])
        return batch

    # processor = Wav2Vec2Processor.from_pretrained("arijitx/wav2vec2-xls-r-300m-bengali")
    processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")

    def prepare_dataset(batch):
        # batch["audio"]["array"] = np.trim_zeros(batch["audio"]["array"], 'fb')
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
        dataset = dataset.map(
            remove_special_characters,
            num_proc=8,
        )
        dataset = dataset.map(
            normalize,
            num_proc=8,
        )
        dataset = dataset.map(prepare_dataset, num_proc=8, remove_columns=dataset.column_names)

        return dataset

    selected_fold = 1

    train_df = df.query(f'fold != {selected_fold}')
    val_df = df.query(f'fold == {selected_fold}')

    train_ds = create_dataset(train_df)
    train_ds.save_to_disk("train")

    val_ds = create_dataset(val_df)
    val_ds.save_to_disk("val")


if __name__ == '__main__':
    create()
