# mobilenetssd-face

## Requirements

- Pytorch (Linux or Mac)

Windows is not working
https://discuss.pytorch.org/t/cant-pickle-local-object-dataloader-init-locals-lambda/31857/8

## Implement face detection using mobilenetssd

### Create dataset

Download fddb and medical-mask-dataset model and extract to each folder. (e.g. /Volumes/ST5/dataset/fddb/)

```
python3 annotation.py mixed /Volumes/ST5/dataset/
```

Output is open-image-dataset format.

```
/Volumes/ST5/dataset/open_images_mixed/sub-test-annotations-bbox.csv
/Volumes/ST5/dataset/open_images_mixed/sub-train-annotations-bbox.csv
/Volumes/ST5/dataset/open_images_mixed/train/images.jpg
/Volumes/ST5/dataset/open_images_mixed/test/images.jpg
```

This is a csv format.

```
ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside,id,ClassName
```

### Train

Download pre-trained model

```
cd pytorch-ssd

wget -P models https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth
```

Train

```
cd pytorch-ssd

python3 train_ssd.py --dataset_type open_images --datasets /Volumes/ST5/dataset/open_images_mixed --net mb2-ssd-lite --pretrained_ssd models/mb2-ssd-lite-mp-0_686.pth --scheduler cosine --lr 0.001 --t_max 100 --validation_epochs 5 --num_epochs 100 --base_net_lr 0.001  --batch_size 5
```

### Test

```
cd pytorch-ssd

python3 run_ssd_example.py mb2-ssd-lite models/mb2-ssd-lite-Epoch-99-Loss-2.466368080392946.pth models/open-images-model-labels.txt ../input.jpg
```

### Convert to onnx

Change convert_to_caffe2_models.py to use opset_version=10.

```
torch.onnx.export(net, dummy_input, model_path, verbose=False, output_names=['scores', 'boxes'], opset_version=10)
```

```
cd pytorch-ssd

python3 convert_to_caffe2_models.py mb2-ssd-lite models/mb2-ssd-lite-Epoch-99-Loss-2.466368080392946.pth models/open-images-model-labels.txt
```
