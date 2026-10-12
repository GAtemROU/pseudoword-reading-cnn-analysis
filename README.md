# Interpretable Analysis of Eye Gaze Behavior During Pseudoword Reading Using Convolutional Neural Networks

This project solves a classification problem by training CNNs on heatmaps and scanpaths from an eye-tracking study (https://dl.acm.org/doi/10.1145/3588015.3590118).

# About
Here is an example of the input images:

<img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/6433069d-b4c7-428d-aa33-73f3c0cc83ae" height='100' width='650'> <img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/8e51b992-1fb6-494f-9bca-b777458b69f2" height='100' width='650'>

The study investigates how one's eye gaze behaves when reading unknown words.

In the original study's paper, logistic regression was used to solve the problem on numerical data. The eye-tracker software also produces heatmaps and scanpaths of eye gaze; this project uses that information.

While this is an unusual approach for such problems, it has some benefits compared to the numerical method.

First, because of the heatmaps, we do not have to choose between saccades and fixations as in the numerical method. These two features are strongly correlated, which suggests using only one of them for machine learning approaches.

Second, this approach is promising in terms of interpretability from a visual perspective. While neural networks are clearly less interpretable than simpler models, visualizing the gradients on images can be a very good tool for interpretability.

# Getting Started
## Data
### Raw Data
The raw data can be downloaded via this link: https://drive.google.com/file/d/16a1fXU_DzUo5mCl-8mpJp-R7JaI086LO/view?usp=drive_link

Please place the files in the data folder. The resulting structure should look as follows:
```Bash
data/rawdata
├── heatmap_fixed
└── scanpaths_fixed
```
### Preprocess
You can find the preprocessing script at `data/preprocessing/preprocess_data.py`.

If you would like to change the cropping arguments, please take a look at `data/preprocessing/data_cropper.py`.

### Data class
The dataset handler is located at `modules/dataset_handling.py`. It helps manage the data, e.g. get a loader or a list of IDs for certain participants.

## Training
Training is done in `training/training.ipynb`.

The class responsible for training is `training/trainer_base.py`. It provides basic functionality such as running epochs and saving models.

### Cross Validation
Due to the relatively small dataset, cross-validation was implemented for training.
`modules/cross_validation.py` is the class responsible for cross-validation.
There are two nested cross-validation loops in the training process. The outer one is used to test the model, while the inner one is used to validate it.

Increasing the `outer_kfolds` parameter increases the number of models trained, as a model is trained from scratch for each outer fold.

Increasing the `inner_kfolds` parameter increases the number of epochs a model is trained for, which corresponds to `num_epochs * inner_kfolds`.

### Logging
Logging is done by `modules/logging.py`. During training, the logs are saved according to the following illustration:

```Bash
experiments/scanpathsb_base_128
├── f1_history
├── log_25.03_1207
├── log_25.03_1212
├── log_25.03_1225
└── log_25.03_1243
```
Here, one log corresponds to one full training run, while the `f1_history` file stores a history of each individual training. Consecutive trainings do not overwrite this information. This is useful when experimenting with hyperparameters.

More general parameters such as type of data, model, or `in_size` are specified in the name of the directory.

The log file looks as follows. After a new fold of test participants is chosen, they are excluded from training.
Then there are statistics for each epoch. After training is done, there are statistics on the test data at the end of the fold.
Confusion matrices are calculated separately for each participant, while accuracy and F1 score take into account samples from all participants in the test set.
```
Fold 1
['10', '11', '13', '14']
Epoch [1], Loss: 19.4181, Val acc: 0.25
Epoch [2], Loss: 8.9236, Val acc: 0.57
...
...
...
Epoch [35], Loss: 0.9230, Val acc: 0.93
Test accuracy: 0.8864, F1 score: 0.8845
Confusion matrices:
{'10': array([[6, 2],
       [2, 1]]), '11': array([[9, 0],
       [0, 2]]), '13': array([[8, 0],
       [0, 3]]), '14': array([[8, 0],
       [1, 2]])}
Fold 2
...
```

### Saving Models
Models are saved under `saved_models/`, followed by a directory that briefly describes the model as well as the time when training started. As discussed above, a separate model is trained for each outer fold.
Hence, a new folder is created for each fold.

If `autosave` is True, intermediate models are also saved during training.
These are the models that reach a new best validation accuracy. In this case, the model is named with the corresponding epoch.

On the other hand, `best_model.pkl` always appears in each fold and represents the first model with the best validation accuracy.

In the example below, `autosave` is True and `outer_kfolds` is 8.
```Bash
saved_models/scanpaths_b_base_22.03_13:39/
├── fold_1
│   ├── best_model.pkl
│   ├── model_epoch_10.pkl
│   ├── model_epoch_11.pkl
│   ├── model_epoch_6.pkl
│   └── model_epoch_8.pkl
├── fold_2
├── fold_3
├── fold_4
├── fold_5
├── fold_6
├── fold_7
└── fold_8
```

Note that if `autosave` is True, `best_model.pkl` and the saved model with the greatest epoch are expected to be identical.

## Architecture
The architecture was kept rather simple.
First, batch normalization without learnable parameters was applied to the input.

Then there are two convolutional layers followed by one fully connected layer.

Both convolutional layers look as follows:
Conv→ReLU→Max Pool→Batchnorm→Dropout

The convolutions have sizes and strides of 3, 1 and 5, 2 respectively.

## Evaluation
To evaluate models more thoroughly, cross-validation was implemented. After training, the model is tested on 4 unseen participants. Then the model is trained from scratch with a different set of test participants. The process is repeated until each participant has been in the test set once.

## Performance
The evaluation mentioned above results in an average F1 score of **0.89**.

## Visualization of gradient
Here is an example of how we can interpret the network using gradients.

Here the sentence did not have any unknown word and the network's prediction was correct. We see that the gradient is distributed across the whole sentence; however, the saturation increases in the middle. This means that the network mostly expected the unknown word to appear in the middle of the sentence. This is a pretty good guess, as the sentences in the original study were constructed in such a way.

<img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/0f955e04-ac4f-4cf8-88df-76eaadd74190" height='400' width='400'><img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/4ee8b179-0dab-4ce3-9cd8-7623eccb1025" height='400' width='400'>

Here the sentence has an unknown word in the center, but the person's behavior is not particularly typical. Instead of rereading the word, they proceed with the sentence to look for clues in other parts of it.

<img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/475b2e5e-01f6-461e-8df9-c4e97feacf5c" height='400' width='400'><img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/15a15a82-80fe-4d89-9cda-946282083833" height='400' width='400'>
