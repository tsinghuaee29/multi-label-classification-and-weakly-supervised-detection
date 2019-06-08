# 训练
  python train_run.py --use_prop_score 1
  参数use_prop_score说明：0: 不添加 use prop_score 数据;
                          1: 添加 prop_score 至 roil_pooling layer;
                          2: 添加 prop_score 至矩阵 det;
                          3: 利用cls的方差约束cls矩阵.
# 测试
  python eval_run.py --use_prop_score 1 --thresh 5
  参数thresh说明：用于筛选候选框的阈值
