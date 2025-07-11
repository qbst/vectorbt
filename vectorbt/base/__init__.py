# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Modules with base classes and utilities for pandas objects, such as broadcasting."""

from vectorbt.base.array_wrapper import ArrayWrapper

# __all__：当使用 from module import xxx 时，xxx 只能是 __all__ 中的元素
__all__ = [
    'ArrayWrapper'
]

__pdoc__ = {k: False for k in __all__}
