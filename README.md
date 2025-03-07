# Transformers Speech Recognition and Accent Classification
## Goal
1. Practice skills learned from the [Hugging Face Transformers for Audio course]( https://huggingface.co/learn/audio-course/en/chapter0/introduction)
2. Explore and implement preexisting transformers models for speech including ASR, phonemic transcription, and accent classification
3. Fine-tune an ASR model for accent classification
4. Demo work on Hugging Face Spaces with [Gradio](https://www.gradio.app/) and create an API endpoint that can take in audio and pass ASR output in json format to a front-end application

## [See demo on Hugging Face Space](https://huggingface.co/spaces/kaysrubio/speech_transcribe_phonemes_and_accent)

## Exploring existing models
I explored Whisper for ASR, several models that provide phonemic transcription using wav2vec2, as well as an existing accent classifier that focused on native English speakers.
 - [openai/whisper-base.en](https://huggingface.co/openai/whisper-base.en) for ASR
 - [vitouphy/wav2vec2-xls-r-300m-timit-phoneme](https://huggingface.co/vitouphy/wav2vec2-xls-r-300m-timit-phoneme) for phonemic transcription trained on DARPA TIMIT for American English
 - [mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme](https://huggingface.co/mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme) for phonemic transcription trained on L2 Arctic dataset with speakers of English as a second language
 - [vitouphy/wav2vec2-xls-r-300m-phoneme](https://huggingface.co/vitouphy/wav2vec2-xls-r-300m-phoneme) for phonemic transcription trained on an unknown dataset
 - [ct-vikramanantha/phoneme-scorer-v2-wav2vec2](https://huggingface.co/ct-vikramanantha/phoneme-scorer-v2-wav2vec2) for phonemic transcription trained on LJSpeech which sounds like Americans reading text in English
 - [Jzuluaga/accent-id-commonaccent_ecapa](Jzuluaga/accent-id-commonaccent_ecapa) for accent classification trained on native English speakers from around the world

## Training my own model for accent classification

### Purpose
The goal of this project is to create an accent classifier for people who learned English as a second language by fine-tuning a speech recognition model to classify accents from 24 people speaking English whose first language is Hindi, Korean, Arabic, Vietnamese, Spanish, or Mandarin.

### Why
Existing accent classifiers focus on native English speakers from around the world but exclude people who learned English as a second language rendering them inaccurate for many common accents among people in the US, such as people whose first language is Spanish or Chinese.

### Data source
The [L2-Arctic](https://psi.engr.tamu.edu/l2-arctic-corpus/) data is ~8GB and comes via email. It includes approximately 24-30 hours of recordings where 24 speakers read passages in English. The first languages of the speakers are Arabic, Hindi, Korean, Mandarin, Spanish, and Vietnamese.  There's 2 women and 2 men in each language group.

### Foundation model
[DistilHuBERT](https://huggingface.co/ntu-spml/distilhubert) is a smaller version of HuBERT that was modified from BERT. BERT is a speech recognition model with encoder-only CTC architecture.  For this project, a classification layer was added.

### Accent classification model I built
DistilHuBERT was fine-tuned on 50% of the L2-Arctic data to classify the accents in the 6 language groups.

The following model was created and uploaded to Hugging Face:
[kaysrubio/accent-id-distilhubert-finetuned-l2-arctic2](https://huggingface.co/kaysrubio/accent-id-distilhubert-finetuned-l2-arctic2)

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 10
- mixed_precision_training: Native AMP

### Limitations
The model is very accurate for novel recordings from the original dataset that were not used for train/test. However, the model is not accurate for voices from outside the dataset.  Unfortunately with only 24 speakers represented, it seems like the model memorized other characteristics of these voices besides accent, thus not creating a model very generalizable to the real world.

### Next Steps
The code is good! If a new dataset becomes available that includes many more voices and clear accent categories, this code may be reused to train a better model.


## Audio Sources Used
  - irish.wav, a clip from [Derry Girls](https://www.youtube.com/watch?v=5J211yVWIzg)
  - indian.wav, a from [Abdul Bari teaching on Algorithms](https://www.youtube.com/watch?v=0IAPZzGSbME&list=PLEouKpnYLW8Gk4w7pe8F5J5UNNIkljZWn)
 - mexican.wav, a clip from Jaime Camil playing Rogelio de la Vega on [Jane the Virgin](https://www.youtube.com/watch?v=7HwnS6R7_wQ)
 - south_african.wav, a clip from [Trevor Noah](https://www.youtube.com/watch?v=xma3ZdwtEJ4)
- chinese-american.wav, a clip from [Ronny Chieng](https://www.tiktok.com/@netflixisajoke/video/7450493571158920478?lang=en)
 - nigerian.wav, a clip from Daniel Etim Effiong and Tana Adelana in [Dinner for Four](https://www.youtube.com/watch?v=QFhI71C4iRI)
 - vietnamese.wav, a clip from the [L2-Arctic](https://psi.engr.tamu.edu/l2-arctic-corpus/) data, participant THV file b0303.wav

