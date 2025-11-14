# 编译器

在异构训练中，编译器是连接算法与多元硬件的核心枢纽。编译器能减少设备间数据传输开销，平衡负载，提升整体效率。通过算子融合、硬件适配等创新，让异构系统发挥最大算力，是驱动 AI 训练性能突破的关键技术支撑。

## DLCompiler

### 介绍
DLCompiler是上海人工智能实验室（上海 AI 实验室）DeepLink 团队开源扩展 Triton 的深度学习编译器：
- 跨架构 DSL 扩展：通过扩展 DSL，让 DSA 芯片（昇腾芯片）也能享受 GPU 级的编程体验和性能，成为 “跨架构 AI Kernel DSL” 。
- 智能自动优化：实现智能核间调度，充分释放多核算力；结合创新的访存合并优化，将离散访问自动重组为高速连续访问，大幅提升算子性能与带宽利用率。

<div align=center>
<img src="https://github.com/user-attachments/assets/59c195cc-2702-4d5a-8559-3bed1722281e" width="50%">
</div>

### 编译使用

#### compile llvm project
```
git clone https://github.com/llvm/llvm-project.git
// triton下的llvm-hash.txt commit id
git reset --hard ed4e505c219fe6c7464ea5a056e90d8cd94c7332

cmake -G Ninja ../llvm  -DLLVM_ENABLE_PROJECTS="llvm;mlir"    -DLLVM_BUILD_EXAMPLES=ON    -DLLVM_TARGETS_TO_BUILD="X86X86;NVPTX;AMDGPU"     -DCMAKE_BUILD_TYPE=Release  -DLLVM_ENABLE_ASSERTIONS=ON       -DLLVM_INSTALL_UTILS=ON

ninja -j64
```


#### 编译 triton && triton
```
export LLVM_BUILD_DIR={path-of-llvm-project}/build
bash compile.sh
export PYTHONPATH=$PWD/third_party/triton/python
export PATH=$PWD/third_party/triton/build/third_party/triton_shared/tools/triton-shared-opt/:$PATH
```


#### 测试
```
cd python/op
python softmax.py
```

#### 刷新code格式
```
bash format.sh
```

### 昇腾芯片
#### 环境准备
准备昇腾设备上环境，可以参考昇腾的链接：https://gitee.com/ascend/triton-ascend
##### 安装ascend cann
1. 要求CANN 版本 > 8.2.RC1.alpha002
2. 社区下载链接：https://www.hiascend.com/developer/download/community/result?module=cann
3. 社区安装指引链接：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit

##### 安装依赖
```
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11
```
##### 安装torch_npu
```
pip install torch_npu==2.6.0rc1
```
#### 编译
```
# set LLVM_INSTALL_PREFIX
bash compile_on_ascend.sh
```

##### ttshared pipeline
```
bash compile_shared.sh apply_patch=true     # 如果不应用patch，可以直接执行 bash compile_shared.sh，如果想要尝试使用新版triton_shared，编译时加上compile_triton_shared=true
# 如果更新了最新版本的triton-shared-opt，需要更新g++版本到11.4，并且手动指定TRITON_SHARED_OPT_PATH:
# conda install -c conda-forge gcc=11.4 gxx=11.4
# export TRITON_SHARED_OPT_PATH=$PWD/third_party/build/triton/build/cmake.linux-aarch64-cpython-3.10/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
```

#### 查看编译过程的mlir文件
```
export DLC_DUMP_IR=1, 默认在当前目录下
```

#### 测试
```
cd python/op
python softmax.py
```

### 寒武纪芯片
#### 编译
```
bash compile_on_mlu.sh
```

#### 测试
```
cd build/triton/tutorials
python 01-vector-add.py
```


## DLBlas

### 总体设计
DLBlas 致力于应用最新技术呈现算子的极致性能，例如ep_moe使用DeepEP、DeepGemm等业界最新技术实现高效的moe模块。

DLBlas 旨在成为一个基于 Triton 的运算符库。因此，内核开发人员可以将其内核注册到该库中，而用户则可以通过提供运算符名称和输入张量来请求运算符。
它通过以下方式改进了 Triton 的自动调谐器:

- **kernel 选择**: 给定相同的运算符，例如 matmul，可能有不同的内核实现；我们希望根据输入张量找到最好的一个。

- **定制配置搜索**: 我们不想枚举所有可能的内核配置（例如 BLOCK_SIZE 等），而是希望使用高级算法（例如贝叶斯优化器）来搜索最佳配置。这需要灵活定义搜索空间和搜索策略。对于 DSA 硬件，配置空间很大。

- **kernel 缓存**：最佳算子实现和内核配置，用于缓存输入张量。其形状、数据类型和设备均特定于特定设备。


### 安装

```
cd dlBLAS
python setup.py install
```
### 开始
有几种方法可以应用 dlblas kernel。
1. 通过get_op导入kernel
```
from dlblas.utils import get_op
args = parse_args()
dtype = torch.float16
device = 'cuda'
a = torch.randn(
    (args.m, args.k),
    dtype=dtype,
    device=device,
)
b = torch.randn(
    (args.k, args.n),
    dtype=dtype,
    device=device,
)
matmul = get_op('matmul', (a, b))
# test
out = matmul(a, b)
ref_out = a @ b
tol = {
    'atol': 1.0,
}
if torch.allclose(out, ref_out, **tol):
    print('✅ Triton and Torch match')
else:
    print('❌ Triton and Torch differ')

```
2. 从kernel文件导入kernel
```
from dlblas.kernels.rms_norm import rms_norm
rms_norm(...)

```
3. 导入 DLBlas 并直接使用
```
import dlblas
dlblas.topk_gating(...)
```
### kernel列表
| Kernel              | API                                                                  |
|:-------------------:|:--------------------------------------------------------------------:|
| silu_and_mul        | from dlblas.kernels.activation import silu_and_mul                   |
| add_rms_norm        | from dlblas.kernels.add_rms_norm import call                         |
| rotary_pos_emb      | from dlblas.kernels.apply_rotary_pos_emb import apply_rotary_pos_emb |
| ffn                 | from dlblas.kernels.ffn import call                                  |
| flash_attention_v2  | from dlblas.kernels.flash_attention_v2 import FlashAttentionV2       |
| fp8_gemm            | from dlblas.kernels.fp8_gemm import fp8_gemm                         |
| fused_rotary_and_fa | from dlblas.kernels.fused_rotary_and_fa import FusedRotaryAndFA      |
| partial_rotary_emb  | from dlblas.kernels.partial_rotary_emb import PartialRotaryEmb       |
| topk_gating         | from dlblas.kernels.topk_gating import TopKGatingFunc                |
