# deepcake
wide and deep model based on tensorflow

官方tutorial
https://www.tensorflow.org/versions/r0.9/tutorials/wide_and_deep/index.html
![](https://www.tensorflow.org/versions/r0.9/images/wide_n_deep.svg)


代码路径s92:/home/serving/yuebin/deepcake， 自己调研请scp走
调研步骤：
1、准备自己的protobuf格式数据(为了读取效率和batch&queue runner支持)
       用csv2pb.py把csv转pb，请提前把csv文件用split命令切块，然后脚本会开多进程转格式。
2、有了pb文件后，在reader.py修改filename_queue和validate_filename_queue





TODO：
1、集群训练，目前worker启动有个bug
2、gpu和cpu模式下效率不高，gpu指定多卡训练时只是用一张卡
