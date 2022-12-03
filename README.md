# End-to-end Entity Linking with Hierarchical Reinforcement Learning

## Overview

This repository contains the Pytorch implementation of paper [[1]](#citation)

Here the [link](https://drive.google.com/file/d/13sMC6IaCFpZKdmr-0KhQu-QmAHR77Dhb/view?usp=sharing) to **pre-processed data** and the **released model**.

## Dependencies

* **python>=3.8**
* **pytorch>=1.7**
* **pytorch_lightning>=1.3**
* **transformers>=4.0**

## Structure
* The [source code](src) of the proposed method. 
* The [model architectures](src/model) of high-level and low-level policy are developed based on [efficient-autoregressive-EL](https://github.com/nicola-decao/efficient-autoregressive-EL/). 
* The entire hierarchical RL [agent](src/hierarchical_el.py) is the composition of the high-level and low-level policy.
* The RL [environment](src/enviroment.py) can be regarded as the dataset module of supervised learning.
* The RL [trainer](src/rl.py) realizes the training process of RL.

## Usage
* run training: python hel_train.py
* run evaluation: python hel_test.py


[1] Lihan Chen, et al. (2023).
End-to-end Entity Linking with Hierarchical Reinforcement Learning.
