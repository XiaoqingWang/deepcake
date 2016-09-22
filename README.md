# deepcake
wide and deep model based on tensorflow
![](https://www.tensorflow.org/versions/r0.9/images/wide_n_deep.svg)

官方提供的wide_n_deep demo有很多问题，比如不支持大量数据的batch加载，调用auc mertic报错，解析csv文件时不支持multi-type column。隐隐感觉tf.contrib是个大坑，所以网上找了一些资料，实现了一个弱化版的wide_n_deep模型，3层nn+lr。

现在特征抽取还比较弱，没加embeding层。但是已经可以载入id类和连续值特征了。id类处理用了hash % bucket。其实也可以用tf.onehot，不过需要知道total-shape。

## 我要训练一个二分类模型
0. 目前特征支持continuous和categorical格式，categorical会用hash bucket进行sparse，暂时不支持feature的cross和embeding。
1. 准备tfrecords格式数据

       **为了支持多线程读取，请用csv2pb.py把自己的csv转tfrecords格式**

       **请提前把csv文件用split命令切块，然后提供的脚本多进程转换格式**

2. 如果需要请修改config中的参数(优化函数啊,learning rate啊什么的)
      **请通过train_pattern， test_pattern指定自己的输入**
3. ``` python cake.py ```
4. 如果要查看模型效果，启动tensorboard

## 坑

1. 目前请不要设置epoch_number，默认为None会循环整个dataset，后续会添加一个early stop策略。目前只能到达理想auc的时候kill掉任务，因为会保存checkpoint，所以不用担心。
2. distribute mode目前worker启动有个bug
3. gpu和cpu模式下效率不高，gpu指定多卡训练时只是用一张卡

官方tutorial
https://www.tensorflow.org/versions/r0.9/tutorials/wide_and_deep/index.html
