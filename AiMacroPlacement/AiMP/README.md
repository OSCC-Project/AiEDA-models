# AiMP

## 介绍
探索macro placement方法

## 软件架构
├── analyses 数据分析模块
├── AutoDMP
├── backend 后端流程，复用eda_engine的流程，可支持iEDA、innovus流程
├── CMakeLists.txt 顶层编译
├── data_manager 数据管理模块，包括AiMP的所有数据（文件路径管理、config、数据结构等），支持顶层Flow数据传递
├── dataset 生成AiMP的数据集
├── eda_engine eda引擎，包含iEDA、innovus、PT等接口
├── __init__.py
├── LICENSE
├── macro_placer 宏单元布局的核心模块，包含宏单元布局流程、核心功能、参数配置等
├── main.py AiMP的入口
├── README.en.md
├── README.md
├── tasks 基于workspace的任务管理模块，支持多任务、多design、多workspace，为达成AiMP性能目标的任务管理
├── third_party 三方库
└── workspace AiMP的工作区资源example，可拷贝后按需求设置


## 安装教程

### 准备环境

首先需要准备总体环境：

- GCC >= 10.3.0
- CMake >= 3.16
- python == 3.10.x
- torch-gpu == 1.11.0

其次确认顶层```CMakeLists.txt```中```PYTHON_EXECUTABLE```的目录，目前167服务器使用的是：

```
/home/zhaoxueyan/anaconda3/envs/iEDA-DSE/bin/python
```

### 编译mt-kahypar

```Shell
# 
$ cd third_party/iEDA/src/third_party/mt-kahypar
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
$ make -j32 
```

### 编译iEDA，并install

```Shell
# 
$ mkdir build
$ cd build
$ cmake ..
$ make install -j32 
```

### 编译AutoDMP，并install

```Shell
# 
$ cd third_party
$ ./install_AutoDMP.sh
```

### 运行AutoDMP

需要先手动屏蔽LEMON_ENABLE_COIN
```
cd third_party/AutoDMP/thirdparty/Limbo/limbo/thirdparty/lemon/CMakeLists.txt

屏蔽下面代码
# IF(LEMON_ENABLE_COIN)
#   FIND_PACKAGE(COIN)
# ENDIF(LEMON_ENABLE_COIN)
```

然后安装autodmp
```Shell
$ cd third_party/AutoDMP/tuner
$ ./run_tuner.sh
```

#### 参与贡献


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
