
# Quick Start

## 仓库下载
如需在硬件芯片中进行计算接口算子实现，可进行以下步骤（具体参考 [DIOPI](https://github.com/DeepLink-org/DIOPI#readme)）。


1. 需下载 [DIOPI仓库](https://github.com/DeepLink-org/DIOPI)，可使用命令：
    ```
    git clone https://github.com/DeepLink-org/DIOPI.git
    ```

    如遇到权限问题，可以参考[FAQ-权限问题]()


## 算子编译

1. 在 DIOPI-IMPL 中新建目录实现 ```DIOPI-PROTO/include/diopi/functions.h``` 声明的标准算子的函数。

    在设备相关目录下提供相应的编译文件，通过以下参考命令进行编译：
    ```
    cd DIOPI-IMPL && sh scripts/diopi_impl.sh torch
    ```

2. 编译 DIOPI-IMPL 提供编译文件的编译计算库。以下示例仅供参考：
    ```
    mkdir build && cd build && cmake .. -DIMPL_OPT=cuda && make -j32
    ```

## 更新基准数据

1. 进入python目录，生成基准数据(需准备 nv 机器和 pytorch1.10 环境)
    ```
    cd python && python main.py --mode gen_data
    ```
    如需指定模型：
    ```
    python main.py --mode gen_data --model_name xxx
    ```
    其中支持的模型名可以通过如下命令获得：
    ```
    python main.py --get_model_list
    ```
2. 将数据拷贝到芯片机器上，执行以下命令验证算子：
    ```
    python main.py --mode run_test
    ```
    如需指定模型：
    ```
    python main.py --mode run_test --model_name xxx
    ```
    如需过滤不支持的数据类型以及部分测试使用nhwc格式张量(如跳过float64以及int64测例)：
    ```
    python main.py --mode run_test --filter_dtype float64 int64 --nhwc
    ```


## 校验算子
1. 将数据拷贝到芯片机器上，执行以下命令验证算子：
    ```
    python main.py --mode run_test
    ```

2. 验证结果分析

    测例通过的输出形式如下：

    ```
    2022-09-29 16:40:40,550 - DIOPI-Test - INFO - Run diopi_functions.relu succeed
    ```
    
    失败的测例会额外存储测例输入参数的张量信息在 ```error_report.csv``` 中以供调试所需。

    ```
    DIOPI-Test Error Report
    ---------------------------------
    1 Tests failed:
    1--Run diopi_functions.batch_norm_backward failed.   TestTag: [float32, backward]  TensorInfo : [(input, float32, (32, 16, 112, 112)), (running_mean, float32, (16,)), (running_var, float32, (16,)), (weight, float32, (16,)), (bias, float32, (16,))]
    ---------------------------------
    Test skipped or op not implemented:
    ```
#### 测例通过
测例通过的输出形式如下：
  ```
  2022-09-29 16:40:40,550 - DIOPI-Test - INFO - Run diopi_functions.relu succeed
  ```
#### 测例失败

失败的测例会额外存储测例输入参数的张量信息在 error_report.csv 中以供调试所需。
  ```
  DIOPI-Test Error Report
  ---------------------------------
  1 Tests failed:
  1--Run diopi_functions.batch_norm_backward failed.   TestTag: [float32, backward]  TensorInfo : [(input, float32, (32, 16, 112, 112)), (running_mean, float32, (16,)), (running_var, float32, (16,)), (weight, float32, (16,)), (bias, float32, (16,))]
  ---------------------------------
  Test skipped or op not implemented:
  ```

