# DIOPI

DIOPI-设备无关算子接口（Device-Independent Operator Interface, DIOPI）在框架和芯片计算库之间定义了统一的**标准接口**。
旨在训练框架和人工智能芯片之间定义了一套计算契约，良好的函数抽象使得上（框架）下（芯片）两层在适配工程实施时能有效地解耦。
基于这套契约训练框架和人工智能芯片可以独立开发，并将下层芯片适配的工作复用到不同的训练框架适配中去，可降低芯片+框架的适配成本，保障算子实现正确性。

其主要的核心功能如下：
1. **<font color="2980b9">提供300+个标准算子接口，包含LLaMa大模型算子接口</font>**。涵盖了大模型、分类、检测、分割及姿态估计等多个领域深度学习模型所需训练算子。
2. **<font color="2980b9">提供统一的标准算子接口，接入7款硬件芯片</font>**。是训练框架和硬件芯片的“桥梁”，降低训练框架和硬件芯片之间的适配成本，创造更好的国产训练生态。
3. **<font color="2980b9">提供标准测试套件，支持11000+个常见算子测例</font>**，为硬件芯片实现的算子库提供调试验证功能。


## 结构说明

![结构](../../_static/image/DIOPI/DIOPI_structure.png)

DIOPI主要包含以下几个组件：

- [**proto**](https://github.com/DeepLink-org/DIOPI/tree/main/proto)：声明了一套运行时函数接口(diopirt)和标准算子接口(function)。
- [**impl**](https://github.com/DeepLink-org/DIOPI/tree/main/impl)：对接硬件芯片。硬件厂商可在其中使用硬件软件栈提供的计算接口，实现算子功能。其使用 ```proto/include/diopi/diopirt.h``` 提供的接口实现 ```proto/include/diopi/functions.h``` 声明的标准算子, 并编译为 ```libdiopi_impl.so``` 动态库。
- [**diopi_test**](https://github.com/DeepLink-org/DIOPI/tree/main/diopi_test)：用于保证算子功能正确性。实现 ```proto/include/diopi/diopirt.h``` 声明基础运行时函数，并调用 ```libdiopi_impl.so``` 进行测试验证。
- [**adaptor**](https://github.com/DeepLink-org/DIOPI/tree/main/adaptor)：用于提供辅助功能函数。目前提供的功能包括自动类型转换、内存分布转换等。

----

### PROTO

PROTO是标准算子接口的原型声明，是芯片厂商实现与框架算子调用的中间层。通过规定标准的运行时函数与算子接口的声明，对框架来说，统一了算子接口，无需考虑芯片厂商具体的算子实现；对厂商来说，可以只聚焦于算子实现与优化，无需考虑框架适配。PROTO作为DIOPI中具体算子声明的环节，起到了承上（框架）启下（芯片厂商）的作用。

PROTO有如下核心功能：
 1. **实现Runtime标准接口定义**。
 声明了在实现标准算子函数时可以使用的工具函数以及相关数据结构。其中，工具函数用于对Context和Tensor两类对象进行操作。
 2. **实现标准算子的接口定义**。
 声明了标准算子的函数，每一个函数完成一个特定的、需要计算设备参与执行的功能。


PROTO的主要组成部分包括 _运行时函数(diopirt)_ 和 _算子声明(functions)_。运行时函数主要为芯片厂商提供实现算子函数时需要框架提供的工具函数，主要包括一些公共类型的声明以及分配与管理张量数据等；算子声明包含了用于人工智能计算的大量函数声明，为各个算子接口的具体参数及其类型提供了标准化的定义；C-API文档生成为算子声明生成API说明文档，供算子开发与使用者查阅更多信息可以查看[PROTO](https://github.com/DeepLink-org/DIOPI/tree/main/proto)。


### IMPL

IMPL 主要用于芯片厂商基于 PROTO 进行标准算子实现，芯片厂商可通过封装自身计算库或者调用 ``kernel`` 的方式来实现 PROTO 定义良好的标准算子接口以备后续测试调用和训练框架调用。

其价值体现在以实现统一接口计算库的形式，来对接不同训练框架。无需考虑不同训练框架特性，可更专注于提升每个功能性算子的性能。更多信息可以查看[IMPL](https://github.com/DeepLink-org/DIOPI/tree/main/impl)。


### DIOPI_TEST

DIOPI_TEST是构建于设备无关算子接口（Device-Independent Operator Interface, DIOPI）之上的测试框架，它支持了没有训练框架的情况下，验证算子适配正确性的功能。DIOPI_TEST设计了一套完整的测试框架和一套算子函数测试。测试套件，可以使芯片厂商适配 DIOPI 算子时，无需训练框架即可对适配结果的正确性进行验证。

主要模块：
* diopi_test 运行时：支持了运行时函数的接口，用以管理设备相关资源。
* 非算子测试：
    * 测试获取设备相关信息标准接口。
    * 测试获取错误信息标准接口。
    * 测试上下文 Context 中 Stream 的正确使用。
* 算子测试：
    * 自定义测例配置：套件提供了描述算子测例的配置文件，用户可自定义扩展测例。
    * 生成基准数据：套件可以根据配置文件生成算子测例的基准输入和输出数据。
    * 校验适配算子：算子适配完成后使用基准输入数据得到的结果与基准输出数据进行比较验证。
* 模型算子测试：
    * 采用算子测试相同的测例配置规则, 使用同一个测试框架生成基准数据并进行测试验证。
    * 从40多个模型训练过程中抓取张量形状，数据类型及其他非张量参数值生成测例。

更多信息可以查看[DIOPI_TEST](https://github.com/DeepLink-org/DIOPI/tree/main/diopi_test)。

### ADAPTER

ADAPTER 是 DIOPI 提供的辅助工具箱，目前提供的功能包括自动类型转换、内存分布转换等，使用时在 IMPL 设备文件夹下添加配置问题，具体配置方法见[IMPL Readme](https://github.com/DeepLink-org/DIOPI/tree/main/impl#readme)。

## Quick Start

### 仓库下载
如需在硬件芯片中进行计算接口算子实现，可进行以下步骤（具体参考 [README](https://github.com/DeepLink-org/DIOPI#readme)）。


1. 需下载 [DIOPI仓库](https://github.com/DeepLink-org/DIOPI)，可使用命令：
    ```
    git clone https://github.com/DeepLink-org/DIOPI.git
    ```

    如遇到权限问题，可以参考[FAQ-权限问题](https://deeplink.readthedocs.io/zh_CN/latest/doc/DIOPI/FAQ.html)


### 算子编译


1. 在设备相关目录下提供相应的编译文件，通过脚本进行编译, 以cuda为例：
    ```
    cd impl && sh scripts/build_impl.sh torch
    ```
    或者参考以下命令示例编译 impl：
    ```
    cd impl && mkdir build && cd build && cmake .. -DIMPL_OPT=torch && make -j32
    ```
### 更新基准数据

1. 进入python目录，生成基准数据(需准备 nv 机器和 pytorch2.0 环境)
    ```
    cd python && python main.py --mode gen_data
    ```
    如需指定模型：
    ```
    python main.py --mode gen_data --model_name xxx
    ```
    其中支持的模型名和对应的算子可以通过如下命令获得：
    ```
    python main.py --get_model_list
    ```
    如果想只生成某一个算子的测例可以使用如下命令, 以add系列的算子为例：
    ```
    python main.py --mode gen_data --fname add
    ```


### 校验算子
1. 将数据拷贝到芯片机器上，执行以下命令验证算子：
    ```
    python main.py --mode run_test
    ```
    如需指定模型：
    ```
    python main.py --mode run_test --model_name xxx
    ```
    如需指定某个算子， 以add为例：
    ```
    python main.py --mode run_test --fname add
    ```
    如需过滤不支持的数据类型以及部分测试使用nhwc格式张量(如跳过float64以及int64测例)：
    ```
    python main.py --mode run_test --filter_dtype float64 int64 --nhwc
    ```
    可以查看[diopi_test Readme](https://github.com/DeepLink-org/DIOPI/tree/main/diopi_test#readme) 了解更详细的设置


2. 验证结果分析

#### 测例通过
测例通过的输出形式如下：
```
2022-09-29 16:40:40,550 - DIOPI-Test - INFO - Run diopi_functions.relu succeed
```

## 常见问题


### 1. DIOPI算子开发流程是怎样的？

- 搭建环境：安装芯片厂商SDK和必要的系统工具。
- 添加算子代码：在impl项目相应目录中添加算子c++代码。
- 生成基准数据：执行基准数据生成命令，生成测试时需要的基准数据。
- 算子测试：执行算子测试命令，将自己实现的算子计算结果和基准数据进行对比。

### 2. 如何搭建IMPL开发环境？如果在自己的PC中开发，需要安装哪些包，cmakelist中include、lib路径需要修改哪些？

首先机器上要有芯片厂商的软件栈，配好环境变量后CMakelist中的include和lib路径第不用修改的，source完环境后可以直接编译。我们推荐使用conda管理python环境，具体安装的包可以在运行报错时，根据提示安装。

### 3. 代码的目录结构是怎样的？编译的命令是什么？编译的结果在哪里？

（1）代码目录结构
* diopi_test主要包含impl(算子实现)，diopi运行时文件和一致性测试的代码
* impl中将不同厂商的的算子实现存放在不同的路径下，例如camb对应寒武纪的算子实现

（2）编译指令
    以寒武纪软件栈为例，先source对应环境, 然后使用如下指令进行编译，
    请注意：对应的软件栈不同，则环境和编译选项也有所不同
```
sh scripts/build_impl.sh camb 
```

（3）编译结果位置
```
/impl/lib下 
```
    
### 4. 生成baseline有哪些环境要求？如何生成baseline并进行测试？生成的数据在哪里？如何查看数据的详细内容？

(1) 生成baseline的环境要求

- ```cuda```：需要环境预装好pytorch，安装pytorch可参考[pytorch官网](https://github.com/pytorch/pytorch)

(2) 如何生成baseline并进行测试？

第一步生成基准输入和输出数据，第二步验证适配的算子的正确性。

测试脚本运行命令（在./python目录下）：
```
python main.py [-h] [--mode MODE] [--fname FNAME]
```
选项说明：
- ```--mode``` 可选项：```gen_data```, ```run_test```
运行模式选项，用于选择当前函数生成基准数据还是测试算子
- ```--fname``` 缺省：```all_ops```
函数名字选项，如果指定函数名字（配置文件中测例的 name）则会对该算子进行基准数据生成和测试，不指定默认对所有算子生成基准数据和测试。

例如：
1.  在 Nvidia 设备上生成基准输入和输出数据
```
python main.py --mode gen_data --fname all_ops
```
2. 在接入芯片设备上运行测试
```
python main.py --mode run_test --fname all_ops
```

(3) 生成的数据在哪里？

在```diopi_test/python/data```中，以pickle形式存储

(4)如何查看数据的详细内容？
有两种方式可以查看数据的详细内容
- ```pickle.load()``` 将测试object读取进内存再进行自定义的可视化和操作，pickle相关使用可以参考[页面](https://docs.python.org/3/library/pickle.html)
- 将```diopi_test/python/conformance/utils.py```中```log_level```设置为```DEBUG```
这样在测试中，如果发现异常（如值不对）则会将数据信息打出

### 5. 如何测试添加的算子是否正确？测试命令是什么？测试结果如何看？如果测试结果不对如何查看更多详细内容？

在README中会有介绍算子测试方法，我们这里使用的是根据```python/conformance/diopi_configs.py```中描述的算子信息在Nvidia机器上生成算子输入以及算子的输出，并将其他芯片厂商的算子运算结果与Nvidia对比。

算子添加后，CI上会进行测试，算子是否正确可看CI日志。测试命令请见README。测试结果会在终端中打印出来。如果结果不正确，可以在```python/conformance/utils.py中将default_cfg_dict[log_level] = DEBUG```。这样会在```python/error_report.csv```中显示详细的错误信息。


### 6. 对于数据类型不支持导致的测试失败如何解决

对于数据类型不支持的测例，提供两种处理方式：

1. 使用ADAPTOR进行类型转换
ADAPTOR可以通过读取设备配置，自动对一些不支持的数据类型进行转换，只需在 impl/ 设备文件夹下添加convert_config.yaml文件，在其中配置不支持的类型及转换规则，编译时即会自动生成转换代码。详细的配置规则参考IMPL的README。

2. 添加设备测试的device_config.py文件（建议放到 impl/ 设备）文件夹下，在其中配置需要跳过的测例以及不支持的数据类型等，使用如下命令运行测试，则会跳过数据类型不支持的测例。device_config.py的详细配置方法参考DIOPI_TEST的README。

```
python main.py --mode run_test --impl_folder device_config.py文件路径。
```

### 7. Clone时出现权限问题？

目前最新的DIOPI仓库中已经没有submodule了，后续如有需要，会在使用教程中补充clone相关步骤。


---
### 无法找到问题
您可在项目中提交issue，将您遇到的问题告诉我们。
<!-- issue回复的流程可在[开发者指南中](Contributors.md)获取。
2. 或者您也可以加入[开发者社区]()，像我们提供反馈和建议。 -->


