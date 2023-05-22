# DIOPI Introduction

DIOPI-设备无关算子接口（Device-Independent Operator Interface, DIOPI）在框架和芯片计算库之间定义了统一的**标准接口**。旨在训练框架和人工智能芯片之间定义了一套计算契约，良好的函数抽象使得上（框架）下（芯片）两层在适配工程实施时能有效地解耦。基于这套契约训练框架和人工智能芯片可以独立开发，并将下层芯片适配的工作复用到不同的训练框架适配中去，可降低芯片+框架的适配成本，保障算子实现正确性。

其主要的核心功能如下：
1. 提供120+个标准算子接口。涵盖了分类、检测、分割及姿态估计等多个领域深度学习模型所需训练算子。
2. 训练框架和硬件芯片的“桥梁”，提供统一的标准算子接口。以此降低训练框架和硬件芯片之间的适配成本，创造更好的国产训练生态。
3. 提供标准测试套件，为硬件芯片实现的算子库提供调试验证功能。


## 结构说明

![结构](../../_static/image/DIOPI_structure.png)


DIOPI主要包含以下几个组件：

- **[DIOPI-PROTO](https://github.com/DeepLink-org/DIOPI/tree/main/DIOPI-PROTO)**：声明了一套运行时函数接口(diopirt)和标准算子接口(function)。
- **[DIOPI-IMPL](https://github.com/DeepLink-org/DIOPI/tree/main/DIOPI-IMPL)**：对接硬件芯片。硬件厂商可在其中使用硬件软件栈提供的计算接口，实现算子功能。其使用 ```DIOPI-PROTO/include/diopi/diopirt.h``` 提供的接口实现 ```DIOPI-PROTO/include/diopi/functions.h``` 声明的标准算子, 并编译为 ```libdiopi_impl.so``` 动态库。在测试阶段，DIOPI-IMPL 还需实现并注册 ```DIOPI-TEST/include/diopi_register.h``` 声明的硬件芯片管理相关的函数。
- **[DIOPI-TEST](https://github.com/DeepLink-org/DIOPI/tree/main/DIOPI-TEST)**：用于保证算子功能正确性。实现 ```DIOPI-PROTO/include/diopi/diopirt.h``` 声明基础运行时函数，并调用 ```libdiopi_impl.so``` 进行测试验证。

----

### DIOPI-PROTO

DIOPI-PROTO是标准算子接口的原型声明，是芯片厂商实现与框架算子调用的中间层。通过规定标准的运行时函数与算子接口的声明，对框架来说，统一了算子接口，无需考虑芯片厂商具体的算子实现；对厂商来说，可以只聚焦于算子实现与优化，无需考虑框架适配。DIOPI-PROTO作为DIOPI中具体算子声明的环节，起到了承上（框架）启下（芯片厂商）的作用。

DIOPI-PROTO的主要组成部分包括 _运行时函数(diopirt）_ 和 _算子声明(functions)_。运行时函数主要为芯片厂商提供实现算子函数时需要框架提供的工具函数，主要包括一些公共类型的声明以及分配与管理张量数据等；算子声明包含了用于人工智能计算的大量函数声明，为各个算子接口的具体参数及其类型提供了标准化的定义；C-API文档生成为算子声明生成API说明文档，供算子开发与使用者查阅

#### 运行时函数(diopirt）
芯片厂商实现的算子函数时，计算过程中可能需要使用设备内存管理、流管理等runtime功能，这些都对设备资源的管理操作需要框架提供相关的函数实现。在DIOPI-PROTO中声明标准的运行时函数抽象，在算子函数运算需要分配内存或要在指定Stream上运行时，即可调用相关函数。
声明的内容主要包括以下部分：
-   错误码```diopiError_t```、数据类型 ```diopiDtype_t```、```diopiSize_t``` 以及不透明数据结构 ```diopiContextHandle_t``` 和 ```diopiTensorHandle_t```；
-   用于对 Tensor 对象进行操作的函数， 包括获取Tensor对象的内存、形状、类型等
-   用于对设备运行上下文Context进行操作的函数，包括获取Stream，构造Tensor对象等
-   其他函数：包括获取当前标准算子接口的版本

#### 算子声明(functions)
目前已实现了约120+个算子的接口声明，涵盖了卷积、归一化、池化、损失函数、基本代数运算、矩阵操作、数学运算等算子类型。


### DIOPI-IMPL

DIOPI-IMPL 主要用于芯片厂商基于 DIOPI-PROTO 进行标准算子实现，和基于 DIOPI-TEST 注册基础少量的运行时函数。芯片厂商需要在 DIOPI-IMPL 通过注册的形式，为后续测试提供如内存拷贝、流创建销毁等可管理设备芯片的功能，该实现部分仅供 DIOPI-TEST 测试所用。更为重要的是，芯片厂商可通过封装自身计算库或者调用 ```kernel``` 的方式来实现 DIOPI-PROTO 定义良好的标准算子接口以备后续测试调用和训练框架调用。

## **实现原理**

* 实现 DIOPI-TEST 所需运行时函数:

  ```DIOPI-TEST/include/diopi_register.h``` 中提供了运行时所需 C-API 函数声明，用户根据函数声明实现运行时所需函数，然后进行注册，以便测试套件能够在芯片上管理内存等资源。该实现部分仅供测试时使用。

* 要求实现并注册的函数列表如下：
  ```
  typedef int32_t (*create_stream_func_t)(diopiStreamHandle_t*);
  //其中diopiStreamHandle_t为void*类型别名;
  typedef int32_t (*destroy_stream_func_t)(diopiStreamHandle_t);

  typedef void* (*malloc_func_t)(uint64_t);
  typedef void (*free_func_t)(void*);

  typedef int32_t (*memcpy_h2d_async_func_t)(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);
  typedef int32_t (*memcpy_d2h_async_func_t)(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);
  typedef int32_t (*memcpy_d2d_async_func_t)(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);

  typedef int32_t (*sync_stream_func_t)(diopiStreamHandle_t stream);

  typedef const char* (*get_last_error_string_func_t)();
  ```
* 实现函数后进行注册：

  实现上述 DIOPI-TEST 所需运行时函数后，通过 DIOPI-TEST/csrc/litert.cpp 提供的注册函数在 initLibrary 中进行注册。示例如下:

  ```
  int32_t initLibrary() {
      // others register function...
      diopiRegisterMemcpyD2DAsyncFunc(cuda_memcpy_d2d_async);
      // others register function...
      return diopiSuccess;
  }
  ```

* 实现 DIOPI 函数接口:

  DIOPI-PROTO/include/diopi/functions.h 根据模型训练和框架开发经验定义了一套标准算子的函数，每一个函数完成一个特定的、需要计算设备参与执行的功能。截止目前，从30个常用模型所需算子的角度出发，定义了所需的常见训练算子。该实现部分会由 DIOPI—TEST 测试后接入训练框架，用于真实模型训练。在实现的过程中，芯片厂商可根据自身特性来优化算子的性能。

  另外，DIOPI-PROTO 提供了如张量，标量等基础数据结构，这些基础数据结构也出现在DIOPI标准算子的参数列表中。而其中一些数据接口如张量 *Tensor*，上下文 *Context* 是不透明数据类型 ***Opaque data type***。 因此 DIOPI-PROTO/include/diopi/diopirt.h 提供了一套接口用以获取 *Tensor* 的相关信息或者从上下文 *Context* 请求资源。这套接口设计旨在连接训练框架和 DIOPI 算子库， 由训练框架提供给 DIOPI 算子库。而 DIOPI-TEST 将以仅为测试服务的原则实现这套接口。





<!--
## Learn More

组件介绍
* [DIOPI-PROTO Readme](https://github.com/DeepLink-org/DIOPI/tree/main/DIOPI-PROTO#readme)
* [DIOPI-IMPL Readme](https://github.com/DeepLink-org/DIOPI/tree/main/DIOPI-IMPL#readme)
* [DIOPI-TEST Readme](https://github.com/DeepLink-org/DIOPI/tree/main/DIOPI-TEST#readme)
<!--* [DIPU-Adapter Readme](DIPU-Adapter.md)-->
<!--
其他文档
<!--* [API文档]{} -->
<!--* [常见问题](https://deeplink-org.github.io/OpenComputeLab.github.io/5%20FAQ.html)
* [Release Note](https://github.com/DeepLink-org/DIOPI/releases)
* [开发者指南](https://github.com/DeepLink-org/DIOPI/blob/main/Contributors.md)

-->