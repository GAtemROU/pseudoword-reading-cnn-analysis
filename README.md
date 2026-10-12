# Interpretable Analysis of Eye Gaze Behavior During Pseudoword Reading Using Convolutional Neural Networks

This project solves classification problem by training CNNs on heatmaps and scanpathes from an eye tracking study (https://dl.acm.org/doi/10.1145/3588015.3590118). 


# About
Here is an example of the input images:

<img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/6433069d-b4c7-428d-aa33-73f3c0cc83ae" height='100' width='650'> <img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/8e51b992-1fb6-494f-9bca-b777458b69f2" height='100' width='650'>

 The study investigates into how ones eye gaze behaves during reading of unknown words.

In the paper of original study, logistic regression was chosen to solve the problem on numerical data. The eye tracker software also produces heatmaps and scanpathes of eye gaze, this information was used in this project.





While this is an unusual approach for such problems, it has some benefits in comparison to numerical method. 

First of all, because of the heatmaps we don't have to choose between saccades and fixations as in the numerical method. These two features are strongly correlated which suggests only using one of them for machine learning approaches. 

Second of all, this approach is promising in terms of interpretability from visual perspective. While neural networks are clearly less interpretable than simpler models, visualizing the gradients on images can be a very good tool for interpretability.

# Getting Started
## Data
### Raw Data
The raw data can be downloaded via the link: https://drive.google.com/file/d/16a1fXU_DzUo5mCl-8mpJp-R7JaI086LO/view?usp=drive_link 

Please locate the files into the data folder, the resulting structure should look as follows:
```Bash
data/rawdata
├── heatmap_fixed
└── scanpaths_fixed
```
### Preprocess
You can find the preprocessing script under the `data/preprocessing/preprocess_data.py`. 

If you would like to change the 
cropping arguments, please take a look at `data/preprocessing/data_cropper.py`

### Data class
Dataset handler is located under `modules/dataset_handling.py`. It helps to manage the data, i.e. get a loader or a list of ids for certain participants. 

## Training
The training is done in the `training/training.ipynb`. 

The class responsible for training is `training/trainer_base.py`. It has basic functionality as running epochs or saving models. 

### Cross Validation
Due to the relatively small data, the cross validation was implemented for training. 
`modules/cross_validation.py` is the class responsible for cross validation.
There are two nested cross validations in the process of training. The outer one is used to test the model, while the inner one is used to evaluate the model.

Increasing the outer_kfolds parameter in the training will increase the number of models trained, as for each outer fold a model is trained from scratch.

Increasing the inner_kfolds parameter in the training will increase the number of epochs a models is trained on, which corresponds to `num_epochs*inner_kfolds`.

### Logging
Logging is done by `modules/logging.py`. During the training process, the logs are saved according to the following illustration:

```Bash
experiments/scanpathsb_base_128
├── f1_history
├── log_25.03_1207
├── log_25.03_1212
├── log_25.03_1225
└── log_25.03_1243
```
Here, one log corresponds to one full training, while the f1_history file has a history of each individual training, 
however consecutive trainings, do not overwrite this information. This is useful when experimenting with hyperparameters.

The more general parameters as type of data, model or the in_size are specified in the name of the directory.

The log file looks as follows, after a new fold the participants test participants are saved, they are not seen during the training at all.
Then we have statistic for each epoch. After the training is done there are statistics on the test data in the end of the fold. 
Confusion matrices are calculated separately for each participant, while accuracy and F1 score takes into account samples from all participants in the test set.
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
The models are saved under the `saved_models/` followed by the directory that shortly represents a models as well as 
the time when the training started. As was already discussed above, each for each outer fold a separate model is trained. 
Hence, a new folder is created for each fold. 

If autosave is True, then the intermediate models are also saved during the training. 
Those are models that reach new best validation accuracy. In this case the model is named with the corresponding epoch.

On the other hand, `best_model.pkl` always appears in each fold. And represents the first model with the best validation accuracy.

In the example below the autosave is True and outer_kfolds is 8. 
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

Note that in case autosave is True, `best_model.pkl` and saved model with the greatest epoch are expected to be identical.

## Architecture 
The architecture was kept rather simple. 
Firstly, batchnormalization without learnable parameters was applied to the input.

Then, there are two convolutional layers followed by one fully connected layer. 

Both convolutional layers look as follows:
Conv->ReLU->Max Pool->Batchnorm->Dropout

The convolitions have sizes and strides of 3, 1 and 5, 2 respectively. 


## Evaluation
To evaluate models better, cross validation was implemented. After training, the model is tested on 4 unseen participants. Then, the model is trained from scratch with a different set of test participants. The process is repeated untill each participant have been to the test set ones. 

## Performance
The evaluation mentioned above results in the average f1 score of **0.89**.

## Visualization of gradient
Here is an example of how we can interpret the network by means of gradient.

Here the sentence did not have any unknown word and the prediction of network was correct. We see that the gradient is distributed among the whole sentence, however, the saturation increases in the middle. Which means that the network was mostly expecting the unkown word to appear in the middle of the sentence. This is a pretty good guess as the sentences in the original study were constructed in such way.

<img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/0f955e04-ac4f-4cf8-88df-76eaadd74190" height='400' width='400'><img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/4ee8b179-0dab-4ce3-9cd8-7623eccb1025" height='400' width='400'>

Here the sentence has an unknown word in the center of it, but the behaviour of the person is not particularly general. Instead of rereading the word, they decide to proceed with the sentence to look for clues in other parts of the sentence.

<img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/475b2e5e-01f6-461e-8df9-c4e97feacf5c" height='400' width='400'><img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/15a15a82-80fe-4d89-9cda-946282083833" height='400' width='400'>


