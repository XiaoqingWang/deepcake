# deepcake
wide and deep model based on tensorflow
一个三层DNN和LR
![](https://www.tensorflow.org/versions/r0.9/images/wide_n_deep.svg)

## 我要训练一个模型

1、准备tfrecords格式数据
       为了支持多线程读取，请用csv2pb.py把自己的csv转tfrecords格式
       **请提前把csv文件用split命令切块，然后提供的脚本多进程转换格式

2、python cake.py --train_pattern=data/pb/x*.train.csv.tfrecords —test_pattern=data/pb/x*.test.csv.tfrecords

BUG&TODO：
0、目前请不要设置epoch_number，默认为None会循环整个dataset，后续会添加一个early stop策略
1、distribute mode目前worker启动有个bug
2、gpu和cpu模式下效率不高，gpu指定多卡训练时只是用一张卡

官方tutorial
https://www.tensorflow.org/versions/r0.9/tutorials/wide_and_deep/index.html
