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

numpy **v2.3.3** 的同一个文件里已经补上了 ``#include <unordered_map>``，
因此这里**只需要把版本号抬高**，不需要再打补丁。

（曾经额外打过一次兜底补丁，但 GNU patch 在 CRLF/LF 混用时对这个文件匹配失败，
反而让构建卡在 "Hunk #1 FAILED at 1"，所以撤掉了：版本覆盖已经足够。）

引用方式见 buildozer.spec 里的 ``p4a.local_recipes``。
"""
from pythonforandroid.recipes.numpy import NumpyRecipe as _BaseNumpy


class NumpyRecipe(_BaseNumpy):
    # v2.3.0 → v2.3.3：修复 unique.cpp 缺少 #include <unordered_map> 的编译错误
    version = "v2.3.3"


recipe = NumpyRecipe()
