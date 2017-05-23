#!/usr/bin/env bash

#训练
#-l1 0.015,0 -l2 0.01,0.005 -r 0.01 -v 10 -t 10000 -r 0.01
bins/mf-train -k 35 -l1 0.015,0 -l2 0,0.05 -t 8000 -r 0.02 data/real_matrix.tr.txt model/libMF_model_l1l2

#预测评分
#bins/mf-predict data/real_matrix.te.txt model/libMF_model output