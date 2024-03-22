
# 新硬件接入

## 新硬件接入PyTorch 2
### 核心代码添加
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

## DICP 实践
### 脚本配置
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

### 单算子测试
示例：加入 `_unsafe_view` 算子
#### 代码添加
需要在 `/home/$USER/code/dicp/dicp/vendor/new_backendGraph/` 自己的后端代码中实现 `_unsafe_view` 算子对应的conversion和codegen等代码（可参考华为和燧原）
- aten算子到相应后端算子的转换
```python
@register_conversion([aten.view.default, aten._unsafe_view, aten._unsafe_view.default])
    def view(self, x, size):
        ...
 ```
- codegen相关代码：因不同后端而异

#### 新建测试文件
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

### 完整模型推理、训练
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