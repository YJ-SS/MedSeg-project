# MedSeg-project

### Current Support Mode
- Dual MBConv VAE ```./seg_method/model/dual_MBConv_VAE.py```
- Simple 3D UNet (MONAI)

## Models to be Supported
- Dual Vision Transformer VAE

## Usage
Maybe you need to creat a floder under ```./training_record``` named model. See 
```config['training_info_config']['model_para_save_path']``` 

Set your configuration in direction ```./train_configuration```, you can change test.json or create a new 
configuration in ```./train_configration``` and named like ```your_config.json```.

Run ```script/train_model.py``` to train dual_MBConv_VAE model with correct json file path.

