<p align="center">
    <img src="resources/reverse-diffusion.gif" alt="drawing" width="500"/>
</p>


# Grad-TTS

Official implementation of the Grad-TTS model based on Diffusion Probabilistic Modelling. For all details check out our paper accepted to ICML 2021 via [this](https://arxiv.org/abs/2105.06337) link.

**Authors**: Vadim Popov\*, Ivan Vovk\*, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov.

<sup>\*Equal contribution.</sup>

## Installation

1: Clone this repository.
2: Install Perl library for French G2P
"""bash
apt-get install libwww-perl -y
"""
3. Install project requirements
"""bash
pip3 install -r new_requirements.txt
"""
4. 
Build `monotonic_align` code (Cython):

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

Note: code is tested on Python==3.6.9.

5. Initialize and Update Submodule for Louisiana Creole TTS
""" 
git submodule init && git submodule update
"""

6. Install customized version of epitran. 
## Inference

You can download Grad-TTS and HiFi-GAN checkpoints trained on LJSpeech dataset (22kHz) from [here](https://drive.google.com/drive/folders/1grsfccJbmEuSBGQExQKr3cVxNV0xEOZ7?usp=sharing).

**Note**: we open-source 2 checkpoints of Grad-TTS. They are the same models but trained with different positional encoding scale: **x1** (`"grad-tts-old.pt"`, ICML 2021 sumbission model) and **x1000** (`"grad-tts.pt"`). To use the former set `params.pe_scale=1` and to use the latter set `params.pe_scale=1000`.

Put necessary Grad-TTS and HiFi-GAN checkpoints into `checkpts` folder in root Grad-TTS directory (note: in `inference.py` you can change default HiFi-GAN path).

1. Create text file with sentences you want to synthesize like `resources/filelists/synthesis.txt`.
2. Run script `inference.py` by providing path to the text file, path to the Grad-TTS checkpoint and number of iterations to be used for reverse diffusion (default: 10):
    ```bash
    python inference.py -f <your-text-file> -c checkpts/grad-tts.pt -t <number-of-timesteps>
    ```
3. Check out folder called `out` for generated audios.

You can also perform *interactive inference* by running Jupyter Notebook `inference.ipynb`.

## Training

1. Make filelists of your audio data like ones included into `resources/filelists` folder. Note, that you need to extract mel-spectrograms and save them on disk as `numpy` files. You can use `mel_spectrogram()` function from `hifi-gan/meldataset.py` to convert your audios into mel-spectrograms.
2. Set experiment configuration in `params.py` file.
3. Specify your GPU device and run training script:
    ```bash
    export CUDA_VISIBLE_DEVICES=YOUR_GPU_ID
    python train.py
    ```
4. To track your training process run tensorboard server on any available port:
    ```bash
    tensorboard --logdir=YOUR_LOG_DIR --port=8888
    ```
    During training all logging information and checkpoints are stored in `YOUR_LOG_DIR`, which you can specify in `params.py` before training.

## References

* HiFi-GAN model is used as vocoder, official github repository: [link](https://github.com/jik876/hifi-gan).
* Monotonic Alignment Search algorithm is used for unsupervised duration modelling, official github repository: [link](https://github.com/jaywalnut310/glow-tts).
* Phonemization utilizes CMUdict, official github repository: [link](https://github.com/cmusphinx/cmudict).
