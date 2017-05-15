#!/bin/bash

#转换成LibFM的text format
./scripts/triple_format_to_libfm.pl -in rat_mat.csv -target 2 -delete_column 3 -separator ","
./scripts/triple_format_to_libfm.pl -in rat_mat_train.csv -target 2 -delete_column 3 -separator ","

#转换成LibFM的bin format
./convert --ifile rat_mat.csv.libfm --ofilex rat_mat.x --ofiley rat_mat.y
./convert --ifile rat_mat_train.csv.libfm --ofilex rat_mat_train.x --ofiley rat_mat_train.y


#转置矩阵
./transpose --ifile rat_mat.x --ofile rat_mat.xt
./transpose --ifile rat_mat_train.x --ofile rat_mat_train.xt


#训练
./libFM -task r -train rat_mat_train -test rat_mat -out predict.txt -rlog log.txt -dim ’1,1,50’
-iter 1000 -method als -regular '0,0,10’ -init_stdev 0.1
./libFM -task r -train ml1m-train.libfm -test ml1m-test.libfm -dim ’1,1,8’ -iter 1000-method mcmc -init_stdev 0.1
./libFM -task r -train ml1m-train.libfm -test ml1m-test.libfm -dim ’1,1,8’ -iter 1000 -method sgda -learn_rate 0.01 -init_stdev 0.1 -validation ml1m-val.libfm
