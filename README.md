# Training
The training code trains new deep networks to simultaneously segment root material in images, and localise key features such as seed locations and tips. Trained models are required as part of the full Modified-RootNav-2 pipeline. A broad overview of the process is this:

1. Train a model to the desired accuracy using a training set, periodically checking accuracy against a separate validation set
2. Optionally verify accuracy against a final test set.
3. Publish the trained model weights and a JSON description to the inference folder.
4. Use the inference code to run Modified-RootNav 2 on new images.

### Training

## Dataset Preparation
Modified-RootNav 2 trains on pairs of images and RSML annotations. RSML can be produced by a number of tools like- (https://sourceforge.net/projects/rootnav/) to do this. Exactly how many images you require will depend on the nature of the images.

## Dataset Directory Format
The dataset should be split into training and validation sets, and an optional testing set. This follows standard convention for the training of deep networks. As an example, assuming your data is stored in a folder called `new_dataset` your folder structure would be as follows:
```
new_dataset/
    train/
    valid/
    test/ [Optional, used after training]
```

Within each folder should be pairs of images and identically named RSML files. when training begins the script will scan the directory for valid training pairs, render segmentation masks and store all required training data within cache files in the same directory.

## Configuration files
Training uses a configuration file to store common hyperparameters, along with directory locations and other information. An example may be found within the training code [here](https://github.com/Kamlesh364/Modified-RootNav2.0/tree/main/training/configs). The majority of this is self explanatory and can be left unchanged. You will need to adapt the dataset path to your folder above. You can also specify transfer learning from a previously trained network, if not the network will train from scratch. We recommend transfer learning from wheat_bluepaper as this is the most established network trained for a long time across over 3,000 images.

## Running Training
Training is run using the following command:
```
python training.py train --config ./path/to/config.yml
```
Optionally you can provide the `--output-example` flag to periodically output an RGB image showing an example segmentation each time the network is validated. This may help when checking the progress of training. Using the above command you will see output like this:
```
Iter [50/25000000]  Loss: 0.4532  Time/Image: 0.1530
Iter [100/25000000]  Loss: 0.3080  Time/Image: 0.1612
Iter [150/25000000]  Loss: 0.1235  Time/Image: 0.1722
Iter [200/25000000]  Loss: 0.0533  Time/Image: 0.1537

...
```
Validation results will also appear here when they are run.
## Testing
The training process will save the best performing network within the run/#### folder. This is the best performance on the validation set, rather than the training set. Despite this, a separate test on new data is worthwhile to ensure the network generalises well. The `test` command can be used to run a single iteration over the test set, providing a number of segmentation and localisation metrics in order to measure performance. Testing is run using the following command:
```
python training.py test --config configs/root_train.yml --model ./new_trained_model.pkl
```
As with training, the config file holds the location of the test set, and the number of threads / batch size. Most other configuration requirements are not relevant to testing. You will see output like this:
```
Processed 50 images

Segmentation Results:
Overall Accuracy: 0.9959
Mean Accuracy:    0.8946
FreqW Accuracy:   0.9987
Root Mean IoU:    0.9645
