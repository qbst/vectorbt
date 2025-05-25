import types

class MethodDescriptor:
    """演示 MethodType 如何与描述符协议配合"""
    
    def __init__(self, func):
        self.func = func
    
    def __get__(self, instance, owner=None):
        if instance is None:
            return self.func  # 从类访问返回函数
        # 从实例访问返回绑定方法 (这就是 MethodType 的作用)
        return types.MethodType(self.func, instance)

def external_function(self, message):
    return f"{self.__class__.__name__}: {message}"

class TestClass:
    # 手动使用描述符
    manual_method = MethodDescriptor(external_function)
    
    def __init__(self, name):
        self.name = name

obj = TestClass("测试")

print("=== 描述符协议中的 MethodType ===")
print(f"从类访问: {type(TestClass.manual_method)}")  # <class 'function'>
print(f"从实例访问: {type(obj.manual_method)}")      # <class 'method'>
print(f"调用结果: {obj.manual_method('Hello')}")