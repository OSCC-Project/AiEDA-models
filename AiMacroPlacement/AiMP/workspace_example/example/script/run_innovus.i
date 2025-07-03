执行innovus步骤如下
1 新建文件夹workspace

2 拷贝文件夹 copy_me_as_a_workspace到workspace并重新命名，作为本次Task的workspace

3 配置definition.tcl，可参考definition_example.tcl
tips： 主要配置4个参数
set DEF_INPUT_PATH ""
set VERILOG_INPUT_PATH ""
set PRE_STEP xxx
set STEP xxx

4 切换到workspace下的run路径
cd <you_workspace_path>/output/innovus/run

5 运行innovus,其中 -log <you_workspace_path>/output/innovus/log/1.log 的1.log名称可以按需要修改
innovus -nowin -init <you_workspace_path>/script/main.tcl -log <you_workspace_path>/output/innovus/log/1.log 

