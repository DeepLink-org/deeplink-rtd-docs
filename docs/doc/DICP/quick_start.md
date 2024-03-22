# Quick Start

标准编译协议（Device-Independent Compile Protocol, DICP）定义了统一的计算描述（中间表示），通过计算图获取深度学习模型中的计算任务表达为上述中间表示，然后通过计算图优化技术自动生成人工智能芯片设备代码，从而提高研发效率和计算的执行性能。

## 准备工作

### 获取代码仓库
```bash
# 拉取DIPU/DICP代码
cd /home/$USER/code
git clone --recurse-submodules https://github.com/DeepLink-org/deeplink.framework.git
```

### 环境配置与编译 DIPU
- 参考 DIPU Quick Start ‒ DeepLink Doc 0.2.2 文档
- 按照上述文档，配置符合DIPU版本要求的Python、GCC等，安装指定的Pytorch，编译DIPU，导出环境变量并验证

### 关联脚本配置
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

### 编译安装DICP
确认 DIPU 安装完成且正常运行后，就可以安装DICP了
```bash
cd /home/$USER/code/dicp
# 面向开发
pip install -e ./
# 面向部署
python setup.py clean && python setup.py install  
```

## 使用DICP
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
