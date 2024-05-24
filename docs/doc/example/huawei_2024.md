# DeepLink实现华为910B上的千卡大规模训练

DeepLink作为芯片与深度学习框架适配桥梁，前端通过deeplink.framework的dipu组件，对接了原生的pytorch深度学习训练框架，底层通过标准化的DIOPI算子接口，适配了华为基于CANN计算框架的Ascend910B人工智能计算专用芯片。针对大模型训练、微调及推理的场景，DeepLink通过DeepLinkExt定义了相关的大模型算子标准化接口，对上适配ModelLink、InternLM、InternEvo等大模型训练框架，对下适配标准化DIOPI定义的大模型算子，赋予Ascend 910B计算平台上高效训练大模型的能力。

当前，DeepLink针对Ascend910B计算平台，已经完成对ModelLink、InternLM、InternEvo等大模型训练框架的适配，并利用这些框架，对上海人工智能实验室开源的书生大模型，商汤科技研发的日日新大模型，Meta公司开源的llama2大模型等进行了千卡的大规模训练。

![Speed compare](../../_static/image/example/example_huawei2024_speed.png)
<p align=center>图1：1024卡上基于DeepLink的torch_dipu和华为torch_npu的对比</p>

以ModelLink+llama2千卡训练为例，当前阶段基于DeepLink + pytorch的大模型训练，相对torch_npu + pytorch，在资源利用率上，前者在host端对cpu的使用需求降低大约5%，在npu设备端，两者对加速芯片的显存使用率和芯片使用率相当。在llama2的训练上，对token的消耗DeepLink + pytorch达到约350 tgs，相对torch_npu + pytorch的345 tgs而言，DeepLink + pytorch方案已经显现出较小的性能优势。

## 适配过程
### 一、环境准备
#### (1) docker镜像。
1.  拉取docker镜像
``` bash
#dipu_latest 指向最新镜像
docker pull registry.sensetime.com/parrots/dipu_ascend:dipu_latest
若要指定版本的镜像
docker pull registry.sensetime.com/parrots/dipu_ascend:dipu0.3.0-a0
#版本号：dipu_latest （根据dipu的tag命名，例如：v0.3.0-a0 对应 dipu0.3.0-a0 ）
```
2. 启动docker镜像
``` bash
docker run -itd \
    -e ASCEND_VISIBLE_DEVICES=6 \
    -p 22133:22 \
    -v /mnt:/mnt \
    --name ${容器名}\
    registry.sensetime.com/parrots/dipu_ascend:dipu_latest bash
    
# ASCEND_VISIBLE_DEVICES npu卡号。选择空闲卡挂载
# -p 端口映射  
``` 

3. 激活DeepLink所需的环境
```bash
source /root/dipu_latest
#或 source /root/dipu0.3.0-a0
```
#### (2) 物理机
```bash
配置Python及gcc工具：
# 准备 python，如 3.9 版本
conda create --prefix=dipu python=3.9
conda activate dipu

# 安装gcc-7.5， 需要root权限
sudo apt-get install gcc-7.5
```

安装cpu版的pytorch
```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### 二、仓库准备
DeepLink适配pytorch和910B芯片进行深度学习模型训练，主要包括deeplink.framework、DIOPI、DeepLinkExt三个组件，其中DIOPI可以独立进行面向底层软件栈的Ascend 910B的适配。

#### (1) DIOPI适配环境
首先拉取DIOPI仓库：
```bash
git clone https://github.com/DeepLink-org/DIOPI.git
```

然后编译目标硬件芯片Ascend 910B的DIOPI算子适配库：
```bash
cd DIOPI/impl
sh scripts/build_impl.sh ascend
```
上述命令默认会把DIOPI编译为Release版本的libdiopi_impl.so库，如需要编译Debug版本的libdiopi_impl.so，可以参考以下命令：

```bash
cd DIOPI/impl
mkdir build
cd build 
cmake .. -DIMPL_OPT=ascend -DCMAKE_BUILD_TYPE=Debug -DTEST=ON 
make -j
```

#### (2) deeplink.framework仓库准备
首先拉取deeplink.framework仓库及其第三方依赖库，主要是[DIOPI](https://github.com/DeepLink-org/DIOPI.git)和[kineto](https://github.com/pytorch/kineto.git)：
```bash
git clone --recurse-submodules https://github.com/DeepLink-org/deeplink.framework.git
```
或者：
```bash
git clone https://github.com/DeepLink-org/deeplink.framework.git
cd deeplink.framework
git submodule update --init --recursive
```
然后编译目标硬件芯片Ascend 910B的deeplink.framework库：
```bash
cd deeplink.framework/dipu
bash scripts/ci/ascend/ci_ascend_script.sh build_dipu
```

#### (3) DeepLinkExt仓库准备
DeepLinkExt组件依赖deeplink.framework和DIOPI组件，需要先根据3.2.2准备好deeplink.framework的编译，才能编译DeepLinkExt组件。

首先拉取DeepLinkExt仓库：
```bash
git clone https://github.com/DeepLink-org/DeepLinkExt
```
然后设置deeplink.framework和DIOPI的环境变量：
```bash
export PYTHONPATH=$WORKDIR/deeplink.framework/dipu/:$PYTHONPATH
export DIPU_ROOT=$WORKDIR/deeplink.framework/dipu/torch_dipu
export DIOPI_PATH=$WORKDIR/deeplink.framework/dipu/third_party/DIOPI/proto
export VENDOR_INCLUDE_DIRS=/usr/local/Ascend/ascend-toolkit/latest/include
```
最后编译DeepLinkExt：
```bash
cd DeepLinkExt
python3 setup.py build_ext --inplace
```
### 三、 910B适配过程
#### (1) 算子适配
DIOPI在模型训练框架和芯片计算库之间定义了统一的[标准算子接口](https://github.com/DeepLink-org/DIOPI/tree/main/proto/include/diopi)，适配Ascend 910B时，Ascend 910的CANN软件栈已经提供了基于AscendCL的底层算子kernel实现。DIOPI适配的工作就是要分析DIOPI算子的定义，及AscendCL kernel的定义及功能，用AscendCL kernel实现DIOPI算子，DIOPI算子实现在impl目录下。

以适配DIOPI的diopiBatchNorm算子为例，首先分析proto中定义的[diopiBatchNorm](https://github.com/DeepLink-org/DIOPI/blob/9c4961ca97c53fd0c4834abe61f05710a8b46985/proto/include/diopi/functions.h#L74)算子，如下：
```c++
/**
 * @brief Applies Batch Normalization for each channel across a batch of data.
 * @param[in] ctx Context environment.
 * @param[in] input input tensor. type = [float32, float16, float64].
 * @param[in] weight weight tensor. type = [float32, float16, float64].
 * @param[in] bias bias tensor. type = [float32, float16, float64].
 * @param[in] running_mean weighted average tensor. type = [float32, float16, float64].
 * @param[in] running_var weighted variance tensor. type = [float32, float16, float64].
 * @param[in] training check if in training mode.
 * @param[in] momentum Used to calculate the running mean and variance during runtime. type = [float32, float64]
 * @param[in] eps The value added to the denominator during batch normalization to ensure numerical stability. type = [float32, float64]
 * @param[out] out normalized result. type = [float32, float16, float64].
 * @param[out] save_mean Mean tensor,the mean value for each feature channel of the input tensor. type = [float32, float16, float64].
 * @param[out] save_invstd Backup of inverse standard deviation computed during training. type = [float32, float16, float64].
 */
DIOPI_API diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                      diopiTensorHandle_t running_mean, diopiTensorHandle_t running_var, bool training, double momentum, double eps);
```

然后根据Ascend 910B 文档分析CANN软件栈中是否有对应的batch_norm的kernel实现，如果CANN软件栈提供了对应的batch_norm的kernel实现，则直接调用该kernel实现；否则，根据CANN软件栈的基本算子能力，在DIOPI中组合实现diopiBatchNorm算子。通过文档可知，cann软件栈提供了aclnnBatchNorm算子kernel，其原型如下：
```c++
aclnnStatus aclnnBatchNormGetWorkspaceSize(const aclTensor *input, 
                                           const aclTensor *weight, 
                                           const aclTensor *bias, 
                                           aclTensor *runningMean, 
                                           aclTensor *runningVar, 
                                           bool training, 
                                           double momentum, 
                                           double eps, 
                                           aclTensor *output, 
                                           aclTensor *saveMean, 
                                           aclTensor *saveInvstd, 
                                           uint64_t *workspaceSize, 
                                           aclOpExecutor **executor)`
aclnnStatus aclnnBatchNorm(void *workspace, 
                           uint64_t workspaceSize, 
                           aclOpExecutor *executor, 
                           const aclrtStream stream)
```
在对齐算子的功能及对应的参数含义后，在DIOPI的impl下实现对应的[diopiBatchNorm算子](https://github.com/DeepLink-org/DIOPI/blob/9c4961ca97c53fd0c4834abe61f05710a8b46985/impl/ascend_npu/diopi_impl/batch_norm.cpp#L14)：
```c++
diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, 
                            diopiTensorHandle_t out, 
                            diopiTensorHandle_t saveMean, 
                            diopiTensorHandle_t saveInvstd,
                            diopiConstTensorHandle_t input, 
                            diopiConstTensorHandle_t weight, 
                            diopiConstTensorHandle_t bias, 
                            diopiTensorHandle_t runningMean,
                            diopiTensorHandle_t runningVar, 
                            bool training, double momentum, double eps) {
    BEGIN_CALL_ACL_OP(out, saveMean, saveInvstd, input, 
                      weight, bias, runningMean, runningVar);
    EXEC_NPU_CMD(aclnnBatchNorm, inputAt, weightAt, biasAt, 
                 runningMeanAt, runningVarAt, training, momentum, 
                 eps, outAt, saveMeanAt, saveInvstdAt);
    END_CALL_ACL_OP();
}
```

算子的适配实现后，还需要设计算子测例，以保证算子功能的正确性，参考 DIOPI的算子校验章节。

#### (2) pytorch适配
DeepLink通过DIOPI标准算子接口接入Ascend 910B后，还需通过dipu对接pytorch的Eager模式，让基于pytorch的模型脚本得以在Ascend 910B平台上进行训练。另外对Graph模式的支持由dicp完成，该部分还在研发中。

dipu结构上分Python和CPP实现两部分，如图。
![图片](../../_static/image/DIPU/structure.png)

dipu的runtime主要分两部分，Core & Distributed和Device。

第一部分Core & Distributed是从pytorch中c10和c10d相关接口中的设备无关部分抽象出来的运行时基类，当前包括`DIPUAllocator`、`DIPUGenerator`、`DIPUStream/Event/Guard`、`ProcessGroupDICL`等。这些类会把设备相关的请求代理到第二部分Device定义的设备相关接口上。

针对Ascend 910B的适配，dipu提供了不同的显存缓存机制，包括BF和BS等缓存策略，大幅提升了显存的利用效率。

第二部分Device是定义的设备相关的接口，不同厂商的芯片对应一组实现。针对Ascend 910B的实现，可以参考[dipu/torch_dipu/csrc_dipu/vendor/ascend/deviceimpl.cpp](https://github.com/DeepLink-org/deeplink.framework/blob/main/dipu/torch_dipu/csrc_dipu/vendor/ascend/deviceimpl.cpp)，其中包含了对设备的管理和显存管理，数据搬运等接口函数的实现。

#### (3) DeepLinkExt
DeepLinkExt对下直接调用DIOPI的算子实现，对上承接了大模型训练、推理框架的算子，并提供了基于pytorch的算子组合实现。

以flash attention算子为例，在ModelLink框架下适配Ascend 910B的主要过程如下。

ModelLink中的[flash attention](https://github.com/Ascend/AscendSpeed/blob/bak/ascendspeed/ops/FlashAttention.py)算子定义为：
``` python
class _FlashAttention(Function):
    @staticmethod
    def forward(ctx, query, key, value, head_num, input_layout, pse, padding_mask, 
                atten_mask, scale, keep_prob, pre_tockens, next_tockens,
                gen_mask_parallel, sync):
        # impl

    @staticmethod
    def backward(ctx, grad_outputs):
        # impl
```
``` python
class FlashAttention(Module):
    def __init__(self):
        super(FlashAttention, self).__init__()
        self.atten = _FlashAttention.apply

    def forward(self, query, key, value, head_num, input_layout, pse=None, 
                padding_mask=None, atten_mask=None, scale=1., keep_prob=1., 
                pre_tockens=2147483647, next_tockens=2147483647, # max of int32
                gen_mask_parallel=True, sync=False):
        # impl
```

分析以上代码可知，[flash attention](https://github.com/Ascend/AscendSpeed/blob/bak/ascendspeed/ops/FlashAttention.py)的实现使用了pytorch的前向和反向计算机制，在DeepLinkExt的接口中，直接对接FlashAttention类，实现其forward和backward成员函数，其[定义及实现](https://github.com/DeepLink-org/DeepLinkExt/blob/main/deeplink_ext/ascend_speed/flash_attention.py)参考如下：

```python
class FlashSelfAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attention_mask, dropout_p, 
                softmax_scale, head_num, input_layout):
        ......
        return out

    @staticmethod
    def backward(ctx, dout):
        return dq, dk, dv, None, None, None, None, None
```

以上在DeepLinkExt中定义好对接ModelLink框架的接口后，在其实现中调用了通过pybind11导出到py空间的DIOPI中定义的flash attention算子。其[定义及绑定](https://github.com/DeepLink-org/DeepLinkExt/blob/33bbb614efafb8f292f73ec9b3b847a653f4c1a7/csrc/extensions.cpp#L195)如下：

```python
auto extFlashAttentionV2(at::Tensor& out, const at::Tensor& q,
                         const at::Tensor& k, const at::Tensor& v,
                         at::Generator& gen, const at::Tensor& attention_mask,
                         double p_dropout, double softmax_scale,
                         int64_t head_num, const std::string& input_layout) {
    ......    
    [[maybe_unused]] auto context = callDiopiKeepContext(
      diopiFlashAttentionV2, out, &dropout_mask, &softmax_max, &softmax_sum,
      &softmax_out, gen, q, k, v, attention_mask, p_dropout, softmax_scale,
      head_num, input_layout.c_str());

    ......
}

auto extFlashAttentionBackward(at::Tensor& grad_q, at::Tensor& grad_k,
                               at::Tensor& grad_v, const at::Tensor& grad_out,
                               const at::Tensor& q, const at::Tensor& k,
                               const at::Tensor& v, const at::Tensor& out,
                               const c10::optional<at::Tensor>& attention_mask,
                               const c10::optional<at::Tensor>& dropout_mask,
                               const at::Tensor& softmax_max,
                               const at::Tensor& softmax_sum,
                               const at::Tensor& softmax_out, double p_dropout,
                               double softmax_scale, int64_t head_num,
                               const std::string& input_layout) {
  callDiopi(diopiFlashAttentionBackward, grad_q, grad_k, grad_v, grad_out, q, k,
            v, out, attention_mask, dropout_mask, softmax_max, softmax_sum,
            softmax_out, p_dropout, softmax_scale, head_num,
            input_layout.c_str());
    ......
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    ......
    if (&diopiFlashAttentionV2 != nullptr) {
        m.def("fa_fwd_v2", &extFlashAttentionV2, "deeplink ext_fa_fwd_v2");
    }
    if (&diopiFlashAttentionBackward != nullptr) {
        m.def("fa_bwd", &extFlashAttentionBackward, "deeplink ext_fa_bwd");
    }
    ......
}
```

上面flash attention的前向计算（extFlashAttentionV2）和反向计算（extFlashAttentionBackward）中调用的diopiFlashAttentionV2和diopiFlashAttentionBackward就是DIOPI中针对flash attention定义的标准算子接口，这两个接口用于适配底层Ascend 910B的算子实现。DIOPI中flash attention算子对Ascend 910B的适配过程可以参考 算子适配章节。

### 四、性能问题解决过程 
#### (1) profiler工具分析热点算子
DeepLink适配好Ascend 910B后，在模型训练过程中，发现性能未达到预期。我们可以借助profiler工具找出热点算子及耗时显著的算子，着重进行算子的性能优化。

以`aten::linalg_vector_norm`算子为例，初期deeplink.framework是通过diopiNorm算子适配的，版本升级后，发现Ascend910B 已经对`aclnnLinalgVectorNorm`进行了独立的支持。DeepLink对该算子进行了快速支持，并通过profiler工具抓取了`aten::linalg_vector_norm`算子的两个版本的性能，实现了算子性能的提升。

