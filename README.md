# kaggle_Bengali.AI_ASR_16th_solution

This is my solution and train code library for
https://www.kaggle.com/competitions/bengaliai-speech

(2023/7/18 ~ 10/18)

detail solution at competition's discussion.
https://www.kaggle.com/competitions/bengaliai-speech/discussion/447965

## train

wav2vec2.0 train code with Transformers.

I trained 2step.

1. easy_dataset

   - using ykg_wer < 0.1.

2. hard_dataset
   - using 0.5 < ykg_wer < 1.5 / ykg_cer < 0.15 / mos_pred > 3

ref:
https://www.kaggle.com/code/imtiazprio/listen-to-training-samples-data-quality-eda

## inference

https://www.kaggle.com/code/neilus/16th-place-solution
