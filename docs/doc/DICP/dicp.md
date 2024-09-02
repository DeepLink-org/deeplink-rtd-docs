# DICP

标准编译协议（Device-Independent Compile Protocol,DICP）定义了统一的计算描述（中间表示），通过计算图获取深度学习模型中的计算任务表达为上述中间表示，然后通过计算图优化技术自动生成人工智能芯片设备代码，从而提高研发效率和计算的执行性能。中间表示是介于源语言和目标语言之间的程序表示，能够极大程度地提高编译流程的可拓展性，同时也能降低优化流程对前端和后端的破坏。多层次中间表示包含从应用到芯片端的多种表示层次，不同层次旨在解决不同尺度的问题。

DICP主要的核心功能如下：
1. **通过接入编译路线带来性能优势，在大模型场景最大限度释放芯片能力**
2. **作为训练框架与国产硬件芯片之间的通用桥梁，支持多种前后端，带来使用易用性**
3. **提供易用、高效的一站式编译适配流程，灵活支持国产硬件图编译器的特性，提高芯片适配效率**

下图描述了DICP在编译链路中的位置：

<div align=center>
<img src="../../_static/image/DICP/dicp_flow.png">
<p>*DICP在编译链路中的位置</p>

</div>

1. 训练框架通过图获取模块将用户的模型代码转换成统一的中间表达。此处的中间表达完全与芯片无关。所以在之后的编译协议部分中，需要建立起与后端芯片的联系。这样才能高效的完成接入。
2. 编译协议完成了衔接框架与芯片编译器的工作，其中包含硬件相关的切图，统一中间表达与芯片所支持的算子之间的映射关系以及数据格式的转换模块。
3. 在编译协议吸收了芯片特点之后，由代码生成模块生成最终的代码，并通过芯片的编译器生成二进制可执行文件之后由框架调用。



## 基于DICP的国产硬件接入PyTorch2实践
<!-- 
### DICP vs 纯Dynamo -->

基于上述DICP，国产硬件可快速接入Pytorch2的编译路线。此路线中的TorchDynamo组件，可使国产硬件在运行时的overhead大幅缩小。  
并且针对国产硬件实现了以下特性：
  - 灵活支持国产硬件图编译器的特性
  - 支持多种国产硬件数据格式
  - 支持动态shape

### 运行逻辑
DICP的运行逻辑如下图所示:
<!-- (**这张图有问题，需要讨论 by jinminxi**) -->

<div align=center>
<img src="../../_static/image/DICP/structure.png">
</div>

其中：
1. **算子映射**： 主要解决框架层算子与后端图编译器的算子之间的语义差别，包括1对1和1对多的转换。  
2. **Shape&Dtype推导**： 进行Shape&data_type的推导，补全整张静态图上的信息，便于之后在代码生成模块能生成代码。  
3. **子图改写**： 将多个小算子融合成为一个或多个适合图编译器的算子，配合后端图编译器将计算效率最大化。
4. **数据格式调整**： 是根据后端芯片与其图编译器的特性，针对特定的算子调整其输入输出的数据格式，使得最大程度的发挥芯片性能。

### 目录结构
* `dicp/dynamo_bridge`： 多后端通用的接入代码，包含了
  1. 接收从AOTAutograd下发而来的FX Graph
  2. 启动各个厂商的IR转换与优化
  3. 启动CodeGen以及JIT缓存的逻辑。
* `dicp/vender`: 主要包含了各个厂商IR的定义，AtenIR到厂商IR的转换，厂商IR上的优化以及最后的代码生成模块。
* `test`: 包含了model测试与op测试


### Demo

#### 安装DICP

```
cd /path_to_dicp
pip install .
```

#### 在华为910上执行llama7B前向推理
```
export DIPU_MOCK_CUDA = false
export DICP_TOPS_DIPU = True
export TEST_DIR = /path_to_dicp/test/
export LLAMA_MODEL_DIR=/path_to_llama_model
bash /path_to_dicp/test/model/run_test_model.sh llama ascendgraph false
```

#### 在燧原T20上执行resnet50训练
```
export DIPU_MOCK_CUDA = false
export DICP_TOPS_DIPU = True
export TEST_DIR = /path_to_dicp/test/
bash /path_to_dicp/test/model/run_test_model.sh resnet50 topsgraph false
```

## Quick Start

标准编译协议（Device-Independent Compile Protocol, DICP）定义了统一的计算描述（中间表示），通过计算图获取深度学习模型中的计算任务表达为上述中间表示，然后通过计算图优化技术自动生成人工智能芯片设备代码，从而提高研发效率和计算的执行性能。

### 准备工作

#### 获取代码仓库
```bash
# 拉取DIPU/DICP代码
cd /home/$USER/code
git clone --recurse-submodules https://github.com/DeepLink-org/deeplink.framework.git
```

#### 环境配置与编译 DIPU
- 参考 DIPU Quick Start ‒ DeepLink Doc 0.2.2 文档
- 按照上述文档，配置符合DIPU版本要求的Python、GCC等，安装指定的Pytorch，编译DIPU，导出环境变量并验证

#### 关联脚本配置
DICP 和 DIPU、 DIOPI 的某些功能可能需要协同或者独立运行，按照实际情况选择配置。
- 设置 DIPU 相关环境变量
```bash
export DIPU_DEVICE=xxx # 设置厂商在 dipu 的设备名，如果在配置DIPU时已经设置好就无需再设置
export DIPU_MOCK_CUDA=false # 是否mock torch.cuda等api
```
- 设置dipoi相关环境变量
```
export DIPU_WITH_DIOPI_LIBRARY=DISABLE # 如果需要禁用diopi，则设置该变量值为DISABLE
```

#### 编译安装DICP
确认 DIPU 安装完成且正常运行后，就可以安装DICP了
```bash
cd /home/$USER/code/dicp
# 面向开发
pip install -e ./
# 面向部署
python setup.py clean && python setup.py install  
```

### 使用DICP
示例：通过昇腾后端 ascendgraph 运行 torch.nn.BatchNorm2d
```python
# 导入torch和dipu相关模块
import os
import torch
import torch_dipu 

# 获取设备
def get_device():
    if os.environ.get("DIPU_MOCK_CUDA") == "True":
        device_name = "cuda"
    else:
        device_name = torch_dipu.dipu.device.__dipu__
    device_index = "0"
    device = f"{device_name}:{device_index}"
    return device

# 定义一个torch Module 或者 一个简单的函数
def foo(a, b, c):
    res = torch.ops.aten.addmm.default(a, b, c)
    return res

device = get_device() # 获取当前设备
backend = 'ascendgraph' # 使用昇腾图编译器后端,目前支持华为昇腾（'ascendgraph'）和燧原T20（'topsgraph'）
dynamic = False # 静态形状输入
compiled_foo = torch.compile(foo, backend=backend, dynamic=dynamic) # TorchDynamo使用指定后端编译模型

dicp_input1 = torch.randn(size=(5,3), dtype=torch.float32).to(device) # 生成输入并转移到设备上
dicp_input2 = torch.randn(size=(5,2), dtype=torch.float32).to(device) # 生成输入并转移到设备上
dicp_input3 = torch.randn(size=(2,3), dtype=torch.float32).to(device) # 生成输入并转移到设备上
dicp_output = compiled_foo(dicp_input1, dicp_input2, dicp_input3) # 得到输出
print(dicp_output.cpu())
```


## 新硬件接入

### 新硬件接入PyTorch 2
#### 核心代码添加
- 在 `/home/$USER/code/dicp/dicp/vendor`中添加新硬件对应的文件夹及相应图编译代码，如 `new_backendGraph/` ，大致代码结构可以参考目前的华为 `AscendGraph/`  和燧原 `TopsGraph/`
```bash
|—— new_backendGraph
|   |—— codegen
|   |   |—— ...
|   |—— __init__.py # 在这里添加下面的代码
|   |—— compile_job.py 
|   |—— config.py
|   |—— conversion.py
|   |—— ...
```
```python
# new_backendGraph/__init__.py
def new_backendgraph(gm, fake_input_tensor):
    from dicp.dynamo_bridge.compile_fx import compile_fx

    return compile_fx(gm, fake_input_tensor, "new_backendgraph")
```
- 在 `/home/$USER/code/dicp/dicp/dynamo_bridge/graph.py` 中添加新硬件的后端模块相关代码
```python
class GraphTransformer:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        backend: str,
    ):
        self.gm = gm
        self.backend = backend
        self.folder = cache_dir()
        self.cpu_gm, self.graph_key = save_cpu_gm(gm, self.folder)
        if backend == 'topsgraph':
            from dicp.vendor.TopsGraph.opset_transform import topsgraph_opset_transform
            self.backend_opset_transform = topsgraph_opset_transform
            from dicp.vendor.TopsGraph.codegen.enflame import EnflameCodegen
            self.backend_codegen = EnflameCodegen
        elif backend == 'ascendgraph':
            from dicp.vendor.AscendGraph.opset_convert import ascendgraph_opset_convert
            self.backend_opset_transform = ascendgraph_opset_convert
            from dicp.vendor.AscendGraph.codegen.ascend import AscendCodegen
            self.backend_codegen = AscendCodegen
        '''
        在这里引入新硬件后端的图变换和代码生成模块
        elif backend == 'new_backendgraph':
            ...
        '''       

        ...
```
- 在 `/home/$USER/code/dicp/setup.py` 加入上面实现的 `new_backendgraph` 入口点
```python
def main():
        
        ...
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Operating System :: POSIX :: Linux"
        ],
        entry_points={
            'torch_dynamo_backends': [
                'topsgraph = dicp.vendor.TopsGraph:topsgraph', # 燧原的入口
                'ascendgraph = dicp.vendor.AscendGraph:ascendgraph', # 昇腾的入口
                'new_backendgraph = dicp.vendor.new_backendGraph:new_backendgraph', # 添加：新后端的入口
                
            ]
        },
        python_requires=">=3.8",
        install_requires=[
            "torch >= 2.0.0a0",
            "torch_dipu == 0.1"
        ]
    )
```

### DICP 实践
#### 脚本配置
- 可以设置不希望custom_fallback的算子
```bash
# 示例，不希望rsqrt.out和_softmax.out custom_fallback
export DIPU_KEEP_TORCHOP_DEFAULT_IMPL_OPS="rsqrt.out,_softmax.out"
```
- 测试的代码在 dicp 的test/ 目录下
```bash
# 设置测试根目录
export TEST_DIR=/home/$USER/code/dicp/test
```

#### 单算子测试
示例：加入 `_unsafe_view` 算子
##### 代码添加
需要在 `/home/$USER/code/dicp/dicp/vendor/new_backendGraph/` 自己的后端代码中实现 `_unsafe_view` 算子对应的conversion和codegen等代码（可参考华为和燧原）
- aten算子到相应后端算子的转换
```python
@register_conversion([aten.view.default, aten._unsafe_view, aten._unsafe_view.default])
    def view(self, x, size):
        ...
 ```
- codegen相关代码：因不同后端而异

##### 新建测试文件
- 使用pytest测试，在 `/home/$USER/code/dicp/test/op` 下新建 `test__unsafe_view.py` 
```python
import pytest
from ..common.utils import (
    torch,
    dynamo,
    parse_args,
    compile_model,
    get_device,
    Size,
    update_dynamo_config,
)

class OpModule(torch.nn.Module):
    def forward(self, a, view_size):
        res_default = torch.ops.aten._unsafe_view.default(a, view_size) # aten算子或者torch接口
        return res_default

model = OpModule()
args = parse_args()
compiled_model = compile_model(model, args.backend, args.dynamic)

class TestUnsafeView():
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("sizes", [Size((5,), (5, 3)), Size((3, 5), (5, 3)), Size((2, 3, 4), (2, 4))])
    @pytest.mark.parametrize("compiled_model", compiled_model)
    def test_torch__unsafe_view(self, sizes, dtype, compiled_model):
        device = get_device() 
        size = sizes.dynamic if compiled_model.dynamic else sizes.static # 动态或静态形状
        input1 = torch.randn(size, dtype=dtype)

        dicp_input1 = input1.to(device) # 需要将数据和模型移动到后端对应设备
        view_size = tuple(reversed(size))

        output = model(input1, view_size)
        dynamo.reset()
        update_dynamo_config(compiled_model.dynamic)
        dicp_output = compiled_model.model(dicp_input1, view_size)

        assert torch.allclose(output, dicp_output.cpu(), equal_nan=True) # 比对设备和cpu上的结果
```
- 在 `/home/$USER/code/dicp/test/new_backend_script/ops` 下的`static.ini` （静态形状测试）、`dynamic.ini` （动态形状测试）中加入 `test__unsafe_view.py`

```python
[pytest]
testpaths = ../../op
python_files = 
                test__log_softmax.py
               test__native_batch_norm_legit_functional.py
               test__adaptive_avg_pool2d.py
               test__adaptive_avg_pool2d_backward.py
               test__softmax.py
               test__unsafe_view.py 
```

- 在 `/home/$USER/code/dicp/test/new_backend_script/ops` 下编写测试脚本 `run_test_ops.sh` 
``` python
if [ ! $TEST_DIR ]; then # 需要前面先定义好测试的根目录
    echo "TEST_DIR is not defined!" >&2
    exit 1
fi
CONFIG_DIR=${TEST_DIR}/new_backend_scripts/ops
TEST_OP_DIR=${TEST_DIR}/op

BACKEND=new_backendgraph # 后端
DYNAMIC=$1 # 动态、静态形状作为命令行参数

CONFIG_STATIC=${CONFIG_DIR}/static.ini 
CONFIG_DYNAMIC=${CONFIG_DIR}/dynamic.ini

cd ${TEST_OP_DIR}
if [ ${DYNAMIC} == false ]; then
    pytest -c ${CONFIG_STATIC} --backend ${BACKEND} --dynamic ${DYNAMIC}
elif [ ${DYNAMIC} == true ]; then
    pytest -c ${CONFIG_DYNAMIC} --backend ${BACKEND} --dynamic ${DYNAMIC}
elif [ ${DYNAMIC} == all ]; then
    pytest -c ${CONFIG_STATIC} --backend ${BACKEND} --dynamic false
    pytest -c ${CONFIG_DYNAMIC} --backend ${BACKEND} --dynamic true
else
    echo "DYNAMIC should in (true, false, all)" >&2
    exit 1
fi
```
- 命令行测试
```bash
bash /home/$USER/code/dicp/test/new_backend_scripts/ops/run_test_ops.sh false
```

#### 完整模型推理、训练
以**mmcv库实现的ResNet50训练**为例，流程类似单个算子
- 需要额外安装mmcv、mmcls库
- 在 `/home/$USER/code/dicp/test/model/test_resnet50.py` 文件已经使用mmcv库完成了网络搭建和训练代码
- 在 `/home/$USER/code/dicp/test/new_backend_script/models` 下的`static.ini` （静态形状测试）、`dynamic.ini` （动态形状测试）中加入 `test_resnet50.py`
```python
[pytest]
testpaths = ../../model
python_files = 
               test_llama.py
               test_stable_diffusion.py
               test_resnet50.py
```
- 在 `/home/$USER/code/dicp/test/new_backend_script/models` 下编写测试脚本 `run_test_models.sh`，与单算子测试的 `run_test_ops.sh` 类似
```python
if [ ! ${TEST_DIR} ] || [ ! ${LLAMA_MODEL_DIR} ] || [ ! ${LLAMA_FINETUNE_DIR} ]; then
    if [ ! ${TEST_DIR} ]; then
        echo "TEST_DIR is not defined!" >&2
    fi
    if [ ! ${LLAMA_MODEL_DIR} ]; then
        echo "LLAMA_MODEL_DIR is not defined!" >&2
    fi
    if [ ! ${LLAMA_FINETUNE_DIR} ]; then
        echo "LLAMA_FINETUNE_DIR is not defined!" >&2
    fi
    exit 1
fi
CONFIG_DIR=${TEST_DIR}/new_backend_scripts/models
TEST_MODEL_DIR=${TEST_DIR}/model

BACKEND=new_backendgraph # 后端
DYNAMIC=$1 # 动态、静态形状作为命令行参数

CONFIG_STATIC=${CONFIG_DIR}/static.ini
CONFIG_DYNAMIC=${CONFIG_DIR}/dynamic.ini

cd ${TEST_MODEL_DIR}
if [ ${DYNAMIC} == false ]; then
    pytest -c ${CONFIG_STATIC} --backend ${BACKEND} --dynamic ${DYNAMIC}
elif [ ${DYNAMIC} == true ]; then
    pytest -c ${CONFIG_DYNAMIC} --backend ${BACKEND} --dynamic ${DYNAMIC}
elif [ ${DYNAMIC} == all ]; then
    pytest -c ${CONFIG_STATIC} --backend ${BACKEND} --dynamic false
    pytest -c ${CONFIG_DYNAMIC} --backend ${BACKEND} --dynamic true
else
    echo "DYNAMIC should in (true, false, all)" >&2
    exit 1
fi
```
- 命令行测试
```bash
bash /home/$USER/code/dicp/test/new_backend_scripts/models/run_test_models.sh false
```