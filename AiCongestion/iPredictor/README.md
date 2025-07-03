# iPredictor

#### Description

- Predicted Metrics:
  - ACC, TNR, FPR, FNR, TPR
    
#### Software Architecture

- data
  - Store the dataset
- DNN
  - dataset_DNN: Defines a class of dataset for DNN model training
  - DNN: DNN model
  - test: Test the DNN model
  - train: Train the DNN model
- logs
  - Record the training process
- model
  - UNet: Semantic segmentation model UNet
- model_run
  - One model corresponds to one model_run 
- utils
  - some functions we often use
- run.py
  - run the model using python
- run.sh
  - run the model using shell

#### Installation

1. No need to install for now

#### Instructions
1. put the data into ./data dir
2. if you want to install mars model
  ```shell
  git clone git://github.com/scikit-learn-contrib/py-earth.git
  cd py-earth
  python setup.py install --cythonize
  ```
3. you can run the model, using python
  ```shell
  run.py -f data/ispd2018/t1/8t1_inst_density_map.csv data/ispd2018/t1/8t1_pin_count_map.csv data/ispd2018/t1/8t1_bbox_congestion_map.csv -l data/ispd2018/t1/8t1_label.csv
  ```
3. you also can run the model, using shell
  ```shell
  bash run.sh
  ```
