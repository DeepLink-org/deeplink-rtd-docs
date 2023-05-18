
# Quick Start

## 硬件芯片适配


如需在硬件芯片中进行计算接口算子实现，可进行以下步骤（具体参考 [DIOPI](https://github.com/DeepLink-org/DIOPI#readme)）。


1. 需下载 [DIOPI仓库](https://github.com/DeepLink-org/DIOPI)，可使用命令：
    ```
    git clone https://github.com/DeepLink-org/DIOPI.git
    ```
2. 在 DIOPI-IMPL 中新建目录实现 ```DIOPI-PROTO/include/diopi/functions.h``` 声明的标准算子的函数。

    在设备相关目录下提供相应的编译文件，通过以下参考命令进行编译：
    ```
    cd DIOPI-IMPL && sh scripts/diopi_impl.sh torch
    ```

## 校验适配算子 <a id="test_tutor"></a>

芯片厂商完成相关算子适配后，可以下载 [DIOPI仓库](https://github.com/DeepLink-org/DIOPI)，并使用如下步骤进行算子正确性验证：

  1. 下载 DIOPI 测验仓库：
      ```
      git clone https://github.com/DeepLink-org/DIOPI.git
      ```
  2. 进入DIOPI-IMPL编译算子实现, 通过以下参考命令进行编译：
      ```
      export DIOPI_BUILD_TESTRT=ON && cd DIOPI-IMPL && sh scripts/diopi_impl.sh torch
      ```
  2. 进入python目录，生成基准数据(需准备nv机器和pytorch1.10环境)

      ```
      cd DIOPI-TEST && python && python main.py --mode gen_data
      ```
      **或** 使用提供的基准测试数据，下载所有数据压缩包，一个MD5SUMS文件。以Mac/Linux系统为例：
      ```
      // 使用md5sum 验证是否准确下载数据
      md5sum -c MD5SUMS

      // 拼接测试数据
      cat diopi_benchmark_data_v1.0.0.tar.gz* >> data.tar

      //解压data。data解压后的位置在：DIOPI-TEST/python/data
      tar -xf data.tar -C your_path_to_DIOPI-TEST/python/
      ```
  3. 将数据拷贝到芯片机器上，执行以下命令验证算子：
      ```
      python main.py --mode run_test
      ```

  4. 验证结果分析

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
