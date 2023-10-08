# 算子分类

## 1. 算子现状
目前深度学习算子数目非常庞杂，据PyTorch研究人员统计，考虑重载、属性区分和别名等复杂因素，PyTorch目前算子总量达到1200个以上。如此数量巨大类型庞杂的算子集合对于多后端芯片适配、编译优化和自动并行等都带来了巨大的挑战。

因此，一个良定义的算子分类，对于整理、分析这些算子非常重要，通过分门别类帮助降低上述复杂度。然而目前分类方法主要是根据算子在PyTorch中的代码组织关系来进行分类的。PyTorch的算子可以从不同层次的Package来调用，例如`torch`, `torch.nn`, `torch.nn.functional`, `torch.Tensor`, `torch.fft`和`torch.linalg`等，它的分类方法基本上基于代码组织，如下图所示（引用自 [Where do the 2000+ PyTorch operators come from?: More than you wanted to know](https://dev-discuss.pytorch.org/t/where-do-the-2000-pytorch-operators-come-from-more-than-you-wanted-to-know/373)）。这样的分类会包含下面几个问题：
  - 缺乏统一的内在分类规则；
  - 分类方法杂糅，而且同一个算子有可能出现在两种分类里；
  - 分类规则缺乏对计算模式的分析；
  - 新增算子很难根据现有分类方法学进行分类。

为此，本文档旨在提出一个分类方法学尽可能地解决上述问题。


## 2. 分类方法学
本文提出的分类方法基于以下4个分类原则：
1. 首先深度学习算子种类尽可能与约定俗成的类型名称保持一致，具体包括：
```
    1. Convolution类
    2. Pooling类
    3. Pad类
    4. Loss类
    5. Norm类
    6. Activation类
    7. Dropout类
    8. Optimizer类
    9. Communication类
    10. Interpolate类
```

2. 算子如果来源于深度学习之外的传统领域，命名与传统领域类型名称保持一致:
```
    1. BLAS类
    2. Linalg类
    3. Permute类，对真实内存做数据重排，典型算子为roll、concat等
    4. View/Copy类，相比Permute类算子，其没有对真实内存数据做操作和移动，只是指针信息或者Copy，包括reshape和Indexing等
    5. Advanced Indexing类
    6. Distribution类
    7. Sort类
```

3. 上述两种类型之外的算子，我们根据计算类型来进行分类，具体地：
我们提出根据计算模式对深度学习算子进行分类的方法。它的核心概念在于输出张量中的每个元素和输入张量中每个输入元素之间的依赖关系。我们首先将输入张量和输出张量展平为一维向量，这样至少有以下两个方面的好处：依赖关系不受具体张量维度和布局影响，依赖关系可以使用一个二维依赖矩阵来表达，每一行为输出张量中的元素，每一列为输入张量中的元素，如果有依赖，对应位置置为1，其余位置置为0。展开为一维向量，与实际物理内存中元素所占内存空间排布类似，更容易反映访存类型。接下来，我们以三种基础的计算展示二维依赖矩阵。
   
   1. `Element-wise类`，它的依赖矩阵的形式为对角矩阵：

    $$ \begin{Bmatrix}
    1&0&0&0\\
    0&1&0&0\\
    0&0&1&0\\
    0&0&0&1
    \end{Bmatrix} $$

                      
    \[ \begin{Bmatrix}
    1&0&0&0\\
    0&1&0&0\\
    0&0&1&0\\
    0&0&0&1
    \end{Bmatrix} \]


   2. `Broadcast类`，它的依赖矩阵为竖线：

    $$\begin{Bmatrix}
    1 & 0 \\
    1 & 0 \\
    0 & 1 \\
    0 & 1
    \end{Bmatrix}$$

   3. `Reduce类`，它的依赖矩阵为横线：

    $$\begin{Bmatrix}
    0 & 0 & 1 & 1\\
    1 & 1 & 0 & 0 \\
    \end{Bmatrix}$$

   4. `Composite类`，由上述三种基础类型组合而成

4. 上述四种类型之外的算子，我们统一称作MISC类


|  算子类型   | 典型算子  |  算子总数  |
|  ----  | ----  | ----  |
| Convolution  | Convolution | 9 |
| Pooling  | max_pooling | 23 |
| Loss  | binary_cross_entropy | 18 |
| Norm  | batch_norm | 12 |
| Activation  | relu | 20 |
| Dropout  | dropout | 10 |
| Optimizer  | sgd | 17 |
| Communication  | all_gather | 15 |
| Interpolate  | grid_sample | 4 |
| BLAS  | mm | 18|
| Linalg  | lu_solve | 34 |
| Permute  | concat、chunck、flip | 13 |
| View/Copy  | squeeze | 18 |
| Advanced Indexing  | index_select | 14 |
| Distribution  | seed | 23 |
| Sort  | topk | 5 |
| Element-wise  | add | 109 |
| Broadcast  | repeat | 18 |
| Reduce  | max | 15 |
| Composite  | addcmul | 7 |
| MISC  | nonzero | 10 |







