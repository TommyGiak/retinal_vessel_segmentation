# README :grimacing:
No seriously, readme, I'm useful :sweat_smile:

## Abstract
This direcory contains a part of a project for the [_Image processing_](https://www.unibo.it/en/teaching/course-unit-catalogue/course-unit/2023/433620) course at UNIBO.\
This project consist in the segmentation of retinal's vessel taken from different patients.\
The dataset used is the _Digital Retinal Images for Vessel Extraction_ (DRIVE), available at [this folder](https://drive.grand-challenge.org/DRIVE/).\
We faced this challenge with two main approaches: machine learnig (U-Net) and classical segmentation techniques (mainly with [Imagej](https://fiji.sc)). In this repo you will find only the machine learnig part of those two.\
The machine learning model is implemented in [_pytorch_](https://pytorch.org) with the support of _torchvision_.

## Repository structure

The repository contain these files/folders:
- _dataset_: folder with the images of the DRIVE database with some preprocessing
- _models_: folder with pytorch pretrained models saved (actually the files contain only the parameters)
- _eval.txt_: imagej macro to evaluate the output of the network with specificity, precision, ...
- _evaluation_results.txt_: evalutaion of the outputs of a pretrained network with 1500 epochs
- _unet_for_the_win.py_: python file where the model is implemented

__N.B.__ The database has 20 images for training (with associated ground truths) and 20 for test (without ground truths). To evaluate the results of our model we divided the training folder in 15 training / 5 test to have groud truths also for the test images.

## Model
The model that we implemented is a U-Net with four encoding layers, scaled with max pooling, followed by four decoder steps with skip connection evry max pooling step.\
Moreover, we decided to add a layer to the rgb channels with the edge detection grayscale image of the rgb one. Therefore, the input channels of the networks are 4 (rgb + edges) and not only 3. The reason for this choice is that this way the network should be facilitated in detecting the smaller vessels, which are also the most challenging to segment.\
Actually in the [_unet_for_the_win.py_](https://github.com/TommyGiak/retinal_vessel_segmentation/blob/main/unet_for_the_win.py) file there are two different networks (with/without edges) with the same structure but one model takes as input the rgb image plus the layer with the graylevel images of the detected edges (4 channels), while the other only the rgb image (3 channels). The models present in the _models_ folder are all trained with the edge layers, with the support of a GPU.\
By the way, I want to thanks [Colab](https://colab.google) for letting us use their GPU, without which the training would have been impossible for us :upside_down_face:

## Install and run the code

To clone this folder, from _terminal_ move into the desired folder and clone this repository using the following command:

```shell
git clone https://github.com/TommyGiak/retinal_vessel_segmentation.git
```

Then I suggest to use an editor like _Spyder_ or _VS Code_ to have the possibility to run different cells indipendently, but if you are bold enough you can run: 

```shell
python unet_for_the_win.py
```

__N.B.__ Remember to adjust the paths if you use an editor and be careful with the names of the file, overwriting and so on...


## Results
Our performance results are shown in the [_evaluation_results.txt_](https://github.com/TommyGiak/retinal_vessel_segmentation/blob/main/evaluation_results.txt) file.\
An example of segmented output of the network, without any thresholding, is show below, in comparison with the input and ground truth.

- input image rgb:
<img src="./datasets/training/images_test/raw/36_training.tif" alt="inp0" width="450"/>
- input edges:
<img src="./datasets/training/edges_test/data/segm_36_training.tif" alt="inp0" width="450"/>
- output of the network trained __with__ edges (without thresholding):
<img src="./datasets/results/result_0.tiff" alt="res0" width="450"/>
- output of the network trained __without__ edges (without thresholding):
<img src="./datasets/results/result_no_edge_0.tiff" alt="res_no0" width="450"/>
- ground truth:
<img src="./datasets/training/1st_manual_test/targets/36_manual1.tif" alt="res0" width="450"/>

That's all for now! :wave:
