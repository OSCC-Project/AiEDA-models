# benchmark
因为服务器原因（155服务器缺库，无法运行加载ieda_py.so），需要将流程engine中iEDA和innovus、pt隔离开，因此将benchmark中需要跑商业工具的模块解耦独立运行。

# run_aimp.py
在166或者167运行aimp的主流程，当运行到innovus和pt模块时，需要切换到155服务器运行，166服务器程序等待155运行结果，成功后继续执行剩余流程。

# run_annalyses.py
可独立运行分析模块

# run_backend.py
运行macro placer后的物理后端流程，可独立支持iEDA、innovus、pt运行



