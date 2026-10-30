# -*- coding: utf-8 -*-
"""覆盖 python-for-android 自带的 numpy recipe。

为什么需要它
------------
p4a master 把 numpy 固定为 **v2.3.0**。这个版本的
``numpy/_core/src/multiarray/unique.cpp`` 只 ``#include <unordered_set>``，
却在文件末尾声明了 ``std::unordered_map<int, function_type> unique_funcs``。
桌面上的 libstdc++ 会顺带把 ``<unordered_map>`` 带进来，所以本地能编过；
Android NDK 用的 libc++ 不会，于是编译直接失败：

    error: no template named 'unordered_map' in namespace 'std';
           did you mean 'unordered_set'?

numpy **v2.3.3** 已补上这个头文件。这里做两件事：
1. 把版本抬到 v2.3.3；
2. 再打一个“确保包含 <unordered_map>”的补丁兜底——即使上游又换回旧标签，
   补丁也能保证头文件存在（重复 include 有 include guard，无害）。

引用方式见 buildozer.spec 里的 ``p4a.local_recipes``。
"""
from pythonforandroid.recipes.numpy import NumpyRecipe as _BaseNumpy


class NumpyRecipe(_BaseNumpy):
    version = "v2.3.3"
    patches = list(getattr(_BaseNumpy, "patches", [])) + ["ensure-unordered-map.patch"]


recipe = NumpyRecipe()
