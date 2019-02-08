# tensorflow_Lab
Testing with tensorflow

# Install tensorflow-gpu
[from here](https://github.com/markjay4k/Install-Tensorflow-on-Ubuntu-17.10-/blob/master/Tensorflow%20Install%20instructions.ipynb)


[Vedio if you want](https://www.youtube.com/watch?v=vxjbL5iN1XY&t=676s)

Also you need to make sure your numpy is new 

# Preparing Inputs

Tensorflow Object Detection API reads data using the TFRecord file format. Two
sample scripts (`create_pascal_tf_record.py` and `create_pet_tf_record.py`) are
provided to convert from the PASCAL VOC dataset and Oxford-IIIT Pet dataset to
TFRecords.


# From tensorflow/models/research/
remeber to do this 

```
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

```

also build and install Setup.py

```
sudo python3 Setup.py build install

```



## Generating the PASCAL VOC TFRecord files.

The raw 2012 PASCAL VOC data set is located
[here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
To download, extract and convert it to TFRecords, run the following commands
below:

```bash
# From tensorflow/models/research/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=train \
    --output_path=pascal_train.record
python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=val \
    --output_path=pascal_val.record
```

You should end up with two TFRecord files named `pascal_train.record` and
`pascal_val.record` in the `tensorflow/models/research/` directory.

The label map for the PASCAL VOC data set can be found at
`object_detection/data/pascal_label_map.pbtxt`.

