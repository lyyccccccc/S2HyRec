# S$^2$HyRec

This is the code for paper:
> S$^2$HyRec: Self-Supervised Hypergraph Sequential Recommendation

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
* Gowalla: SNAP: Network datasets: Gowalla (stanford.edu)
* Movielens: MovieLens 10M Dataset | GroupLens
* Yelp: https://www.yelp.com/dataset


## Usage
Use the following command to train the S$^2$HyRec on the Gowalla dataset:

* python main.py --data gowalla --lr 2e-3 --reg 1e-2 --temp 0.1 --save_path gowalla --epoch 150 --batch 512 --graphNum 3 --gnn_layer 2 --att_layer 2  --alpha 0.6 --trend_loss_weight 1e-4  --test True --testSize 1000 --ssldim 48


  
