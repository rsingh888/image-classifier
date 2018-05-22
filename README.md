# Image classifer to predict flower name from image

## Training the model
### Basic Usage
  **python train.py data_directory**
  
  Prints out training loss as the network trains

##### Options:
Set directory to save checkpoints: **python train.py data_dir --save_dir save_directory**

Choose architecture: **python train.py data_dir --arch "vgg13"**

Set hyperparameters: **python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20**

Use GPU for training: **python train.py data_dir --gpu**

## Prediction from already trained model
  Predict flower name from an image with predict.py along with the probability of that name
### Basic usage: 
  **python predict.py input checkpoint**

#### Options:
Return top KK most likely classes: **python predict.py input checkpoint --top_k 3**

Use a mapping of categories to real names: **python predict.py input checkpoint --category_names cat_to_name.json**

Use GPU for inference: **python predict.py input checkpoint --gpu**
