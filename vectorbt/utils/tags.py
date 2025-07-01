# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)
import ast

from vectorbt import _typing as tp
from vectorbt.utils.template import RepEval


def match_tags(tags: tp.MaybeIterable[str], in_tags: tp.MaybeIterable[str]) -> bool:
    """
    标签匹配函数。
    tags 是一个字符串列表，每个字符串是一个布尔表达式，in_tags中出现过的项替换为True，否则替换为False
    替换并使用RepEval运算后，tags中至少有一个为True，则返回True
    
    参数说明：
        tags (tp.MaybeIterable[str]): 要匹配的标签或标签列表
            - 可以是单个字符串或字符串列表
            - 支持简单标签（如'hello'）或布尔表达式（如'hello and world'）
        in_tags (tp.MaybeIterable[str]): 目标标签集合
            - 可以是单个字符串或字符串列表  
            - 必须全部为有效的Python标识符
    
    返回值：
        bool: 如果任一标签匹配成功则返回True，否则返回False
    
    使用示例：
        >>> from vectorbt.utils.tags import match_tags
        
        # 基础标签匹配
        >>> match_tags('hello', 'hello')
        True
        >>> match_tags('hello', 'world') 
        False
        
        # 多标签匹配（OR逻辑）
        >>> match_tags(['hello', 'world'], 'world')
        True
        >>> match_tags('hello', ['hello', 'world'])
        True
        
        # 布尔表达式匹配
        >>> match_tags('hello and world', ['hello', 'world'])
        True
        >>> match_tags('hello and not world', ['hello', 'world'])
        False
        >>> match_tags('strategy1 or (momentum and not reversal)', ['strategy1', 'momentum'])
        True
    """
    if isinstance(tags, str):
        tags = [tags]
    if isinstance(in_tags, str):
        in_tags = [in_tags]
    
    # 验证in_tags中的所有标签都是有效的Python标识符：
    # 1.只能包含字母、数字和下划线
    # 2.不能以数字开头
    # 3.不能是Python保留字
    # 4.区分大小写
    for in_t in in_tags:
        if not in_t.isidentifier():
            raise ValueError(f"Tag '{in_t}' must be an identifier")

    # 遍历每个要匹配的标签
    for t in tags:
        # 检查当前标签是否为简单标识符
        if not t.isidentifier():
            # 如果不是标识符，则作为布尔表达式处理
            
            # 使用AST解析表达式，提取所有变量名
            # ast.walk()遍历AST树的所有节点
            # ast.Name类型的节点表示变量引用
            node_ids = [node.id for node in ast.walk(ast.parse(t)) if type(node) is ast.Name]
            
            # 创建求值映射：将表达式中的变量名映射为布尔值
            # 如果变量名在in_tags中存在则为True，否则为False
            eval_mapping = {id_: id_ in in_tags for id_ in node_ids}
            
            # 使用RepEval安全地执行布尔表达式
            # RepEval是vectorbt的模板评估工具，提供安全的表达式执行环境
            eval_result = RepEval(t).eval(eval_mapping)
            
            # 验证表达式求值结果必须是布尔值
            if not isinstance(eval_result, bool):
                raise TypeError(f"Tag expression '{t}' must produce a boolean")
            
            # 如果表达式求值为True，则匹配成功
            if eval_result:
                return True
        else:
            # 如果是简单标识符，直接检查是否在目标标签集合中
            if t in in_tags:
                return True
    
    # 如果所有标签都不匹配，返回False
    return False
