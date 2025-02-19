# S2HyRec

This is the code for paper:
> S2HyRec: Self-Supervised Hypergraph Sequential Recommendation



!["S2HyRec Performance Comparison"](https://github.com/lyyccccccc/S2HyRec/blob/main/S2HyRec_framework.png)



## Dependencies
Recent versions of the following packages for Python 3 are required:
* matplotlib==3.3.4
* pandas==1.1.5
* scipy==1.5.2
* numpy==1.17.0
* tensorflow_gpu==1.15.0

## Datasets
All of the datasets we use are publicly available datasets.
### Link
The used datasets are available at:
* Gowalla: https://snap.stanford.edu/data/loc-Gowalla.html
* Movielens: https://grouplens.org/datasets/movielens/10m/
* Yelp: https://www.yelp.com/dataset


## Usage
Use the following command to train the S2HyRec on the Gowalla dataset:

* python main.py --data gowalla --lr 5e-3 --reg 1e-2 --temp 0.1 --save_path gowalla --epoch 150 --batch 512 --graphNum 3 --gnn_layer 1 --att_layer 1  --alpha 0.6 --ssl_weight 1e-6  --hyperNum 256 --test True --testSize 1000 --pos_length 300


  
