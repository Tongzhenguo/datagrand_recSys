#!/bin/bash

#转换成LibFM的text format
bins/scripts/triple_format_to_libfm.pl -in data/trainData.csv,data/testData.csv -target 2  -separator ","

#转换成LibFM的bin format
bins/convert --ifile data/trainData.csv.libfm --ofilex data/trainData.x --ofiley data/trainData.y
bins/convert --ifile data/testData.csv.libfm --ofilex data/testData.x --ofiley data/testData.y

#转置矩阵
bins/transpose --ifile data/trainData.x --ofile data/trainData.xt
bins/transpose --ifile data/testData.x --ofile data/testData.xt


#训练
bins/libFM -task r -train data/trainData -test data/testData -out data/predict.txt  -dim '1,1,8’
#-iter 10 -method als -regular '0,0,10’ -init_stdev 0.1
#bins/libFM -task r -train ml1m-train.libfm -test ml1m-test.libfm -dim '1,1,8’ -iter 1000-method mcmc -init_stdev 0.1
#bins/libFM -task r -train ml1m-train.libfm -test ml1m-test.libfm -dim '1,1,8’ -iter 1000 -method sgda -learn_rate 0.01 -init_stdev 0.1 -validation ml1m-val.libfm


