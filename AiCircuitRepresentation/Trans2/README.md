# Trans^2: Transformer-Based Circuit Representation Learning and Transformer-Based Optimization Representation


## Abstract 
  Transformer models have recently gained popularity in graph representation learning because of their potential to learn complex relationships beyond the ones captured by regular graph neural networks (GNNs). However, current circuit representation learning methods hardly utilize its representative capability. Therefore, without the knowledge of high-order structural information, current representation methods can be easily affected by simple optimization transformations performed on the circuits. In this paper, we propose \textit{Trans$^2$} framework with several adaptations on the transformer to enhance the representation capabilities over optimized circuits: (1) The motif-based encoding as initial node embeddings and several tasks supervised on motif distribution, (2) An aggregation function with the attention mechanism that is more efficient and compatible with circuits, and (3) the OPTrans that enables the embeddings to represent the optimized circuits structurally and functionally. Trans$^2$ improves the robustness and soundness against optimization transformation to the circuit. Our experiments on general tasks demonstrate significant improvements compared with other popular SOTA methods, proving the effectiveness of Trans$^2$. 

## Installation
```sh
conda create -n trans2 python=3.8.18
conda activate trans2
pip install -r requirements.txt
```

## Model Training 
1. Prepare Dataset
cd data
```
wget https://github.com/Ironprop-Stone/python-deepgate/releases/download/dataset/train.zip
unzip train.zip 
cd ..
python ./src/prepare_dataset.py --exp_id train --aig_folder ./dataset/rawaig # for original dataset
```

2. Model Training
The model training is separated into two stages. DAGTrans and OPTrans.
```sh
bash ./run/stage1_train.sh
bash ./run/stage2_train.sh
```


