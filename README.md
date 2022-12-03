# End-to-end Entity Linking with Hierarchical Reinforcement Learning

## Overview

This repository contains the Pytorch implementation of paper [[1]](#citation)

## Dependencies

* **python>=3.8**
* **pytorch>=1.7**
* **pytorch_lightning>=1.3**
* **transformers>=4.0**

## Structure
* The [source code](src) of the proposed method. 
* The [model architectures](src/model) of high-level and low-level policy are developed based on [efficient-autoregressive-EL](https://github.com/nicola-decao/efficient-autoregressive-EL/). 
* The entire hierarchical RL [agent](src/hierarchical_el.py) is the composition of the high-level and low-level policy.
* The RL [environment](src/environment.py) can be regarded as the dataset module of supervised learning.
* The RL [trainer](src/rl.py) realizes the training process of RL.

## Evaluation instruction
* Download and decompress the [pre-processed data and released model](https://drive.google.com/file/d/13sMC6IaCFpZKdmr-0KhQu-QmAHR77Dhb/view?usp=sharing) in the root directory of the repository .
* Run evaluation of the released model:

  $ python hel_test.py
* Expected results:

	| model          | micro_f1_sdm       | ... |
	|----------------|--------------------|-----|
	| rl_model.torch | 0.8756436945327323 | ... |

## Training instruction
* Download the [pre-trained model](https://drive.google.com/)
* Train a new model based on the pre-trained model (assumed [pretrain_path] is its path): 

  $ python hel_train.py --pretrained [pretrain_path]

[1] Lihan Chen, et al. (2023).
End-to-end Entity Linking with Hierarchical Reinforcement Learning.
