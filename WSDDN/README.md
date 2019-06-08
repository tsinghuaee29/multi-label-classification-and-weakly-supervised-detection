# WSDDN.pytorch

PyTorch implementation of [Weakly Supervised Deep Detection Network](<https://arxiv.org/pdf/1511.02853.pdf>), CVPR 2016.

## Train

 ```sh
 python train_run.py --use_prop_score 1
 ```

### Options

- `use_prop_score`
  - `0` : not using use prop_score 
  - `1` : use prop_score data for roi_pooling layer
  - `2` : use prop_score data for the matrix det
  - `3` : use prop_score data for variance limitation

## Test

```sh
  python eval_run.py --use_prop_score 1 --thresh 5
```

### Options

- `thresh` : threshold for proposal selection



## Reference

This project is based on https://github.com/deneb2016/WSDDN.pytorch