# Welcome to BiLSTM+Debias
---
## Introduction

This source code is the basis of the following paper:
> Learning when to trust distant supervision: An application to low-resource POS tagging using cross-lingual projection, CoNLL 2016

## Building
It's developed on clab/cnn toolkit.
- Install clab/cnn following [clab/cnn](https://github.com/clab/cnn-v1).
- Add the source code to folder *cnn/examples* and add ``bilstm-dn`` and ``tag-bilstm-dn`` to *CMakeLists.txt*.
- Make again.

## Data format
The format of input data is as follows:
```
Tok_1 Tok_2 ||| Tag_1 Tag_2
Tok_1 Tok_2 Tok_3 ||| Tag_1 Tag_2 Tag_3
...
```

## How train a model
```sh
./bilstm-dn gold_data_file projected_data_file dev_file test_file max_epochs [model.params]
```
The algorithm exports the best parameters to ``model.params`` and the best results to ``best_params.txt``. You can continue training the model from the last best parameters by including ``model.params`` as the last argument.

## How to tag a text using the trained model
```sh
./tag-bilstm-dn gold_data_file projected_data_file model.params file_to_tag output
```
**TODO:** Tag file without including the ``gold_data_file`` and ``projected_data_file`` as arguments (currently you need the ``gold_data_file`` and ``projected_data_file`` that you train the model with in order to be able to tag a file).
