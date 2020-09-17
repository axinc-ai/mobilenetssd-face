# mobilenetssd-face

## Implement face detection using mobilenetssd

### test

```
cd pytorch-ssd
wget -P models https://storage.googleapis.com/models-hao/gun_model_2.21.pth
wget -P models https://storage.googleapis.com/models-hao/open-images-model-labels.txt
python3 run_ssd_example.py mb1-ssd models/gun_model_2.21.pth models/open-images-model-labels.txt gun.jpg
```

```
masked
half_masked
no_mask
```

### download test data

```
python3 open_images_downloader.py --root ~/data/open_images --class_names "Handgun,Shotgun" --num_workers 1
```

train-annotations-bbox.csv

```
ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside,id,ClassName
c220bdb28a6c6506,xclick,/m/0gxl3,1,0.503268,0.75,0.627451,0.960784,0,0,0,0,0,/m/0gxl3,Handgun
```

### train

```
wget -P models https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
python3 train_ssd.py --dataset_type open_images --datasets ~/data/open_images --net mb1-ssd --pretrained_ssd models/mobilenet-v1-ssd-mp-0_675.pth --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 100 --base_net_lr 0.001  --batch_size 5
```

### test

```
python3 run_ssd_example.py mb1-ssd models/mobilenet-v1-ssd-Epoch-99-Loss-2.2184619531035423.pth models/open-images-model-labels.txt ~/Downloads/gun.JPG
```


