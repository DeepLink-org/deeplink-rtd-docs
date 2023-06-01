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

:::info

Some **content** with _Markdown_ `syntax`. Check [this `api`](#).

:::

> 注释
> PEP 8 的代码规范并不是绝对的，项目内的一致性要优先于 PEP 8 的规范。OpenMMLab 各个项目都在 setup.cfg 设定了一些代码规范的设置，请遵照这些设置。一个例子是在 PEP 8 中有如下一个例子：
```
# Correct:
hypot2 = x*x + y*y
# Wrong:
hypot2 = x * x + y * y
```
> 这一规范是为了指示不同优先级，但 OpenMMLab 的设置中通常没有启用 yapf 的 ARITHMETIC_PRECEDENCE_INDICATION 选项，因而格式规范工具不会按照推荐样式格式化，以设置为准。
