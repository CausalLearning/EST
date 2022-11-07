# Expression Snippet Transformer for Robust Video-based Facial Expression Recognition

Pytorch implementation of paper: 

> **Expression Snippet Transformer for Robust Video-based Facial Expression Recognition **

## Content

- [Dependencies](#dependencies)
- [Code and Data Preparation](#code-and-data-preparation)
- [Training](#training)
- [Testing](#testing)

## Dependencies

Please make sure Python>=3.6 is installed

Required packages are listed in requirements.txt. You can install them by running:

```
pip install -r requirements.txt
```

## Code and Data Preparation

1. Download from this repository and download  the pre-trained ResNet-18 from [Baidu Drive](https://pan.baidu.com/s/1lnO1alaaP23NlZcPyNOhgg) (1req)

2. Prepare the dataset.

   You need to unified the input video length to 105 frames. Make sure the data structure is as below.

   ```
   ├── DFEW
      └── 14400
          ├── 001.jpg
          ├── 002.jpg
          ├── 003.jpg
          ├── ...
      └── 14401
          ├── 001.jpg
          ├── 002.jpg
          ├── 003.jpg
          ├── ...
   ```

## Training

You can use the following command to train:

```
python main.py --train_video_root /data/Your_Path/data_path/DFEW/Data_105img --train_list_root /data/Your_Path/data_path/train_test_set/Train_DFEW_all_clip_set_2.txt --test_video_root /data/Your_Path/data_path/DFEW/Data_105img --test_list_root /data/Your_Path/data_path/train_test_set/Test_DFEW_all_clip_set_2.txt --dataset_name DFEW --name dfew_transformer --gpu_ids 3 --batch 8 --epochs_count 160
```

## Testing

You can evaluate a trained model by running:

```
python main.py --train_video_root /data/yuanyuan/fcx/WECLNet_per/data_path/DFEW/Data_105img --train_list_root /data/yuanyuan/fcx/WECLNet_per/data_path/train_test_set/Train_DFEW_all_clip_set_1.txt --test_video_root /data/yuanyuan/fcx/WECLNet_per/data_path/DFEW/Data_105img --test_list_root /data/yuanyuan/fcx/WECLNet_per/data_path/train_test_set/Test_DFEW_all_clip_set_1.txt --dataset_name DFEW --name dfew_transformer --gpu_ids 3 --batch 8 --phase test --eval_model_path MODEL_PATH
```

Here, `MODEL_PATH` denotes for the path of the trained model.

You can download our trained model on DFEW from [Baidu Drive](https://pan.baidu.com/s/1BkZnt5IP-xcXcSiTlcuKsA) (owu2)
