# 介绍
探索 AI 放置宏单元的方法

# 数据流
## 输入
verilog文件，工艺环境相关文件
## 输出
### innovus
输出摆放宏单元位置的tcl script，替换默认的innovsu脚本 ${top_name}_macro_loc.tcl


## 1. Run AutoDMP

### 1.1 Build AutoDMP

```Shell
$ pwd
123
$ mkdir build
123
$ ./tuner/run_tuner.sh 0 1 test/ariane133_nangate45/configspace.json test/ariane133_nangate45/ariane.aux test/ariane133_nangate45/ariane133_ppa.json \"\" 20 2 0 0 10 ./tuner test/ariane133_nangate45/mobohb_log

```