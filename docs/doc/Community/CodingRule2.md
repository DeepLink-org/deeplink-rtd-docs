# 代码规范

## 代码规范标准

### PEP 8 —— Python 官方代码规范

[Python 官方的代码风格指南](https://peps.python.org/pep-0008/)，包含了以下几个方面的内容：

* 代码布局，介绍了 Python 中空行、断行以及导入相关的代码风格规范。比如一个常见的问题：当我的代码较长，无法在一行写下时，何处可以断行？

* 表达式，介绍了 Python 中表达式空格相关的一些风格规范。

* 尾随逗号相关的规范。当列表较长，无法一行写下而写成如下逐行列表时，推荐在末项后加逗号，从而便于追加选项、版本控制等。

```
    # Correct:
    FILES = ['setup.cfg', 'tox.ini']
    # Correct:
    FILES = [
        'setup.cfg',
        'tox.ini',
    ]
    # Wrong:
    FILES = ['setup.cfg', 'tox.ini',]
    # Wrong:
    FILES = [
        'setup.cfg',
        'tox.ini'
    ]
```

* 命名相关规范、注释相关规范、类型注解相关规范，我们将在后续章节中做详细介绍。

“A style guide is about consistency. Consistency with this style guide is important. Consistency within a project is more important. Consistency within one module or function is the most important.” PEP 8 – Style Guide for Python Code

Some **content** with _Markdown_ `syntax`. Check [this `api`](#).


> 注释
> PEP 8 的代码规范并不是绝对的，项目内的一致性要优先于 PEP 8 的规范。OpenMMLab 各个项目都在 setup.cfg 设定了一些代码规范的设置，请遵照这些设置。一个例子是在 PEP 8 中有如下一个例子：

``` python
    # Correct:
    hypot2 = x*x + y*y
    # Wrong:
    hypot2 = x * x + y * y
```
> 这一规范是为了指示不同优先级，但 OpenMMLab 的设置中通常没有启用 yapf 的 ARITHMETIC_PRECEDENCE_INDICATION 选项，因而格式规范工具不会按照推荐样式格式化，以设置为准。

## Google 开源项目风格指南

[Google 使用的编程风格指南](https://google.github.io/styleguide/pyguide.html)，包括了 Python 相关的章节。相较于 PEP 8，该指南提供了更为详尽的代码指南。该指南包括了语言规范和风格规范两个部分。

其中，语言规范对 Python 中很多语言特性进行了优缺点的分析，并给出了使用指导意见，如异常、Lambda 表达式、列表推导式、metaclass 等。

风格规范的内容与 PEP 8 较为接近，大部分约定建立在 PEP 8 的基础上，也有一些更为详细的约定，如函数长度、TODO 注释、文件与 socket 对象的访问等。

推荐将该指南作为参考进行开发，但不必严格遵照，一来该指南存在一些 Python 2 兼容需求，例如指南中要求所有无基类的类应当显式地继承 Object, 而在仅使用 Python 3 的环境中，这一要求是不必要的，依本项目中的惯例即可。二来 OpenMMLab 的项目作为框架级的开源软件，不必对一些高级技巧过于避讳，尤其是 MMCV。但尝试使用这些技巧前应当认真考虑是否真的有必要，并寻求其他开发人员的广泛评估。

另外需要注意的一处规范是关于包的导入，在该指南中，要求导入本地包时必须使用路径全称，且导入的每一个模块都应当单独成行，通常这是不必要的，而且也不符合目前项目的开发惯例，此处进行如下约定：

``` python
    # Correct
    from mmcv.cnn.bricks import (Conv2d, build_norm_layer, DropPath, MaxPool2d,
                                Linear)
    from ..utils import ext_loader

    # Wrong
    from mmcv.cnn.bricks import Conv2d, build_norm_layer, DropPath, MaxPool2d, \
                                Linear  # 使用括号进行连接，而不是反斜杠
    from ...utils import is_str  # 最多向上回溯一层，过多的回溯容易导致结构混乱
```

OpenMMLab 项目使用 pre-commit 工具自动格式化代码，详情见 [贡献者指南](https://deeplink.readthedocs.io/zh_CN/latest/doc/Community/Contributors.html)。

## 命名规范

### 命名规范的重要性

优秀的命名是良好代码可读的基础。基础的命名规范对各类变量的命名做了要求，使读者可以方便地根据代码名了解变量是一个类 / 局部变量 / 全局变量等。而优秀的命名则需要代码作者对于变量的功能有清晰的认识，以及良好的表达能力，从而使读者根据名称就能了解其含义，甚至帮助了解该段代码的功能。

### 基础命名规范

<center>

<table>
<tr>
<th >类型</th>
<th>公有</th>
<th>私有</th>
</tr>
<tr>
<td>模块</td>
<td>lower_with_under</td>
<td>_lower_with_under</td>
</tr>
<tr>
<td>包</td>
<td>lower_with_under</td>
<td>_lower_with_under</td>
</tr>
<tr>
<td>类</td>
<td>CapWords</td>
<td>_CapWords</td>
</tr>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
</table>

</center>


注意：

* 尽量避免变量名与保留字冲突，特殊情况下如不可避免，可使用一个后置下划线，如 class_

* 尽量不要使用过于简单的命名，除了约定俗成的循环变量 i，文件变量 f，错误变量 e 等。

* 不会被用到的变量可以命名为 _，逻辑检查器会将其忽略。

### 命名技巧

良好的变量命名需要保证三点：

* 含义准确，没有歧义
* 长短适中
* 前后统一

``` python
    # Wrong
    class Masks(metaclass=ABCMeta):  # 命名无法表现基类；Instance or Semantic？
        pass

    # Correct
    class BaseInstanceMasks(metaclass=ABCMeta):
        pass

    # Wrong，不同地方含义相同的变量尽量用统一的命名
    def __init__(self, inplanes, planes):
        pass

    def __init__(self, in_channels, out_channels):
        pass
```

常见的函数命名方法：

* 动宾命名法：crop_img, init_weights
* 动宾倒置命名法：imread, bbox_flip

注意函数命名与参数的顺序，保证主语在前，符合语言习惯：

* ``check_keys_exist(key, container)``
* ``check_keys_contain(container, key)``

注意避免非常规或统一约定的缩写，如 ``nb -> num_blocks``，``in_nc -> in_channels``


## 注释规范

### 为什么要写注释

对于一个开源项目，团队合作以及社区之间的合作是必不可少的，因而尤其要重视合理的注释。不写注释的代码，很有可能过几个月自己也难以理解，造成额外的阅读和修改成本。

### 如何写注释

最需要写注释的是代码中那些技巧性的部分。如果你在下次代码审查的时候必须解释一下，那么你应该现在就给它写注释。对于复杂的操作，应该在其操作开始前写上若干行注释。对于不是一目了然的代码，应在其行尾添加注释。 —— Google 开源项目风格指南

``` python
    # We use a weighted dictionary search to find out where i is in
    # the array. We extrapolate position based on the largest num
    # in the array and the array size and then do binary search to
    # get the exact number.
    if i & (i-1) == 0:  # True if i is 0 or a power of 2.
```

为了提高可读性, 注释应该至少离开代码2个空格. 另一方面, 绝不要描述代码. 假设阅读代码的人比你更懂Python, 他只是不知道你的代码要做什么. —— Google 开源项目风格指南

``` python
    # Wrong:
    # Now go through the b array and make sure whenever i occurs
    # the next element is i+1

    # Wrong:
    if i & (i-1) == 0:  # True if i bitwise and i-1 is 0.
```

在注释中，可以使用 Markdown 语法，因为开发人员通常熟悉 Markdown 语法，这样可以便于交流理解，如可使用单反引号表示代码和变量（注意不要和 docstring 中的 ReStructured 语法混淆）

``` python
    # `_reversed_padding_repeated_twice` is the padding to be passed to
    # `F.pad` if needed (e.g., for non-zero padding types that are
    # implemented as two ops: padding + conv). `F.pad` accepts paddings in
    # reverse order than the dimension.
    self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
```

### 注释示例

1. 出自 ``mmcv/utils/registry.py``，对于较为复杂的逻辑结构，通过注释，明确了优先级关系。

``` python
    # self.build_func will be set with the following priority:
    # 1. build_func
    # 2. parent.build_func
    # 3. build_from_cfg
    if build_func is None:
        if parent is not None:
            self.build_func = parent.build_func
        else:
            self.build_func = build_from_cfg
    else:
        self.build_func = build_func
```

1. 出自 ``mmcv/runner/checkpoint.py``，对于 bug 修复中的一些特殊处理，可以附带相关的 issue 链接，帮助其他人了解 bug 背景。

``` python
    def _save_ckpt(checkpoint, file):
        # The 1.6 release of PyTorch switched torch.save to use a new
        # zipfile-based file format. It will cause RuntimeError when a
        # checkpoint was saved in high version (PyTorch version>=1.6.0) but
        # loaded in low version (PyTorch version<1.6.0). More details at
        # https://github.com/open-mmlab/mmpose/issues/904
        if digit_version(TORCH_VERSION) >= digit_version('1.6.0'):
            torch.save(checkpoint, file, _use_new_zipfile_serialization=False)
        else:
            torch.save(checkpoint, file)
```

## 类型注解

### 为什么要写类型注解

类型注解是对函数中变量的类型做限定或提示，为代码的安全性提供保障、增强代码的可读性、避免出现类型相关的错误。 Python 没有对类型做强制限制，类型注解只起到一个提示作用，通常你的 IDE 会解析这些类型注解，然后在你调用相关代码时对类型做提示。另外也有类型注解检查工具，这些工具会根据类型注解，对代码中可能出现的问题进行检查，减少 bug 的出现。 需要注意的是，通常我们不需要注释模块中的所有函数：

1. 公共的 API 需要注释
2. 在代码的安全性，清晰性和灵活性上进行权衡是否注释
3. 对于容易出现类型相关的错误的代码进行注释
4. 难以理解的代码请进行注释
5. 若代码中的类型已经稳定，可以进行注释. 对于一份成熟的代码，多数情况下，即使注释了所有的函数，也不会丧失太多的灵活性.

### 如何写类型注解

1. 函数 / 方法类型注解，通常不对 self 和 cls 注释。

``` python
    from typing import Optional, List, Tuple

    # 全部位于一行
    def my_method(self, first_var: int) -> int:
        pass

    # 另起一行
    def my_method(
            self, first_var: int,
            second_var: float) -> Tuple[MyLongType1, MyLongType1, MyLongType1]:
        pass

    # 单独成行（具体的应用场合与行宽有关，建议结合 yapf 自动化格式使用）
    def my_method(
        self, first_var: int, second_var: float
    ) -> Tuple[MyLongType1, MyLongType1, MyLongType1]:
        pass

    # 引用尚未被定义的类型
    class MyClass:
        def __init__(self,
                    stack: List["MyClass"]) -> None:
            pass
```

注：类型注解中的类型可以是 Python 内置类型，也可以是自定义类，还可以使用 Python 提供的 wrapper 类对类型注解进行装饰，一些常见的注解如下：

``` python
    # 数值类型
    from numbers import Number

    # 可选类型，指参数可以为 None
    from typing import Optional
    def foo(var: Optional[int] = None):
        pass

    # 联合类型，指同时接受多种类型
    from typing import Union
    def foo(var: Union[float, str]):
        pass

    from typing import Sequence  # 序列类型
    from typing import Iterable  # 可迭代类型
    from typing import Any  # 任意类型
    from typing import Callable  # 可调用类型

    from typing import List, Dict  # 列表和字典的泛型类型
    from typing import Tuple  # 元组的特殊格式
    # 虽然在 Python 3.9 中，list, tuple 和 dict 本身已支持泛型，但为了支持之前的版本
    # 我们在进行类型注解时还是需要使用 List, Tuple, Dict 类型
    # 另外，在对参数类型进行注解时，尽量使用 Sequence & Iterable & Mapping
    # List, Tuple, Dict 主要用于返回值类型注解
    # 参见 https://docs.python.org/3/library/typing.html#typing.List
```

1. 变量类型注解，一般用于难以直接推断其类型时

``` python
    # Recommend: 带类型注解的赋值
    a: Foo = SomeUndecoratedFunction()
    a: List[int]: [1, 2, 3]         # List 只支持单一类型泛型，可使用 Union
    b: Tuple[int, int] = (1, 2)     # 长度固定为 2
    c: Tuple[int, ...] = (1, 2, 3)  # 变长
    d: Dict[str, int] = {'a': 1, 'b': 2}

    # Not Recommend：行尾类型注释
    # 虽然这种方式被写在了 Google 开源指南中，但这是一种为了支持 Python 2.7 版本
    # 而补充的注释方式，鉴于我们只支持 Python 3, 为了风格统一，不推荐使用这种方式。
    a = SomeUndecoratedFunction()  # type: Foo
    a = [1, 2, 3]  # type: List[int]
    b = (1, 2, 3)  # type: Tuple[int, ...]
    c = (1, "2", 3.5)  # type: Tuple[int, Text, float]
```

3. 泛型

上文中我们知道，typing 中提供了 list 和 dict 的泛型类型，那么我们自己是否可以定义类似的泛型呢？

```python
    from typing import TypeVar, Generic

    KT = TypeVar('KT')
    VT = TypeVar('VT')

    class Mapping(Generic[KT, VT]):
        def __init__(self, data: Dict[KT, VT]):
            self._data = data

        def __getitem__(self, key: KT) -> VT:
            return self._data[key]
```

使用上述方法，我们定义了一个拥有泛型能力的映射类，实际用法如下：

``` python
    mapping = Mapping[str, float]({'a': 0.5})
    value: float = example['a']
```

另外，我们也可以利用 TypeVar 在函数签名中指定联动的多个类型：

``` python
    from typing import TypeVar, List

    T = TypeVar('T')  # Can be anything
    A = TypeVar('A', str, bytes)  # Must be str or bytes


    def repeat(x: T, n: int) -> List[T]:
        """Return a list containing n references to x."""
        return [x]*n


    def longest(x: A, y: A) -> A:
        """Return the longest of two strings."""
        return x if len(x) >= len(y) else y
```

更多关于类型注解的写法请参考 [typing](https://docs.python.org/3/library/typing.html)。

### 类型注解检查工具

mypy 是一个 Python 静态类型检查工具。根据你的类型注解，mypy 会检查传参、赋值等操作是否符合类型注解，从而避免可能出现的 bug。

例如如下的一个 Python 脚本文件 test.py:

``` python
    def foo(var: int) -> float:
        return float(var)

    a: str = foo('2.0')
    b: int = foo('3.0')  # type: ignore
```

运行 mypy test.py 可以得到如下检查结果，分别指出了第 4 行在函数调用和返回值赋值两处类型错误。而第 5 行同样存在两个类型错误，由于使用了 type: ignore 而被忽略了，只有部分特殊情况可能需要此类忽略。

``` python
    test.py:4: error: Incompatible types in assignment (expression has type "float", variable has type "int")
    test.py:4: error: Argument 1 to "foo" has incompatible type "str"; expected "int"
    Found 2 errors in 1 file (checked 1 source file)
```


