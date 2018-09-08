# Sigmedia-AVSR

Audio-Visual Speech Recognition (AVSR) research system using sequence-to-sequence neural networks based on TensorFlow

# About

Sigmedia-AVSR is an open-source research system for Speech Recognition, developed by the Sigmedia team
in Trinity College Dublin, Ireland.

Written entirely in Python, Sigmedia-AVSR aims to provide a simple and reproducible way of training and evaluating
speech recognition models based on sequence to sequence neural networks. Sigmedia-AVSR can exploit both auditory
and visual speech modalities, considered either independently (ASR, VSR) or together (AVSR).  

Rather than providing a dense documentation to the users and contributors, the Sigmedia-AVSR code is designed
(or strives) to be intuitive and self-explanatory, encouraging researchers and developers to understand the entire
codebase and propose improvements at its lowest levels. Hence we want it be more of a flexible research system than
a black box for production. For didactic purposes, please refer to Sigmedia-ASR, which is the single modality
precursor written in a more compact form and no longer maintained.


# Core functionalities

#### 1. Extract acoustic features from audio files (librosa, TensorFlow)
* log mel-scale spectrograms, MFCC
* optional computation of first and second derivatives
* optional strided frame stacking
* write into TensorFlow-compatible format (TFRecord dataset)
    
#### 2. Extract the lip region from video files (OpenFace - Tadas Baltrusaitis)
* write into TensorFlow-compatible format (TFRecord dataset)

#### 3. Train sequence to sequence neural networks for continuous speech recognition
* audio-only (LAS [3])
* visual-only (lip-reading [5])
* audio-visual fusion
    * dual-attention decoder (WLAS [4])
    * attention-based alignment (AV-Align [6])
* flexible language units (phonemes, visemes, characters etc.)
 
#### 4. Evaluate models
* normalised Levenshtein distances
    * Character Error Rate
    * Word Error Rate

# Getting started

A typical workflow is as follows:

1. convert data into .tfrecord files
2. train/evaluate models

Please refer to the attached examples for running audio-only, visual-only, or audio-visual speech recognition experiments.

To prepare the data, you can use the two scripts `extract_faces.py` and `write_records_tcd.py`

For faster prototyping, we recommend checking out our publicly available audio-visual dataset, [TCD-TIMIT](https://sigmedia.tcd.ie/TCDTIMIT/)

# Dependencies

For visual/audio-visual experiments, please compile from source install [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)

The other dependencies are popular and easy to install Python packages, so feel free to use your preferred sources.
Unless otherwise stated, all dependencies should be kept updated to their latest stable versions to avoid compatibility issues.


# Acknowledgements

We are grateful to Eugene Brevdo of Google for his remarkable help and advice during the early stages of Sigmedia-ASR 
(precursor of Sigmedia-AVSR) development. In addition, we would like to thank 
Derek Murray, Andreas Steiner, Khe Chai Sim for the assistance and interesting conversations, and also every
TensorFlow contributor on GitHub and StackOverflow.

# How to cite

If you use this work, please cite it as:

George Sterpu, Christian Saam, and Naomi Harte. 2018. 
Attention-based Audio-Visual Fusion for Robust Automatic Speech Recognition. 
In 2018 International Conference on Multimodal Interaction (ICMI ’18), October 16–20, 2018, Boulder, CO, USA.
ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3242969.3243014

[bib tba, arxiv avail]

# References

[1]: Sequence to Sequence Learning with Neural Networks
https://arxiv.org/abs/1409.3215
    
[2] Neural Machine Translation by Jointly Learning to Align and Translate
https://arxiv.org/abs/1409.0473

[3] Listen, Attend and Spell
https://arxiv.org/abs/1508.01211

[4] Lip Reading Sentences in the Wild
https://arxiv.org/abs/1611.05358

[5] Can DNNs Learn to Lipread Full Sentences?
https://arxiv.org/abs/1805.11685

[6] Attention-based Audio-Visual Fusion for Robust Automatic Speech Recognition
https://arxiv.org/abs/1809.01728

# Contact

George Sterpu sterpug [at] tcd.ie  
Dr. Naomi Harte nharte [at] tcd.ie
