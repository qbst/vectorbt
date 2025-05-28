# 编写一个类中静态方法的例子
class MyClass:
    @staticmethod
    def static_method():
        print("This is a static method")
# 验证 __new__ 是描述符
print(type(MyClass.static_method))  # <class 'builtin_function_or_method'>
print(hasattr(MyClass.static_method, '__get__'))  