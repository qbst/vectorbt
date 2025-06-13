class MyDescriptor:
    def __init__(self):
        pass
    def __get__(self, instance, owner):
        return "描述符返回值"
    
def user_func(x):
    pass

user = MyDescriptor()

print(type(MyDescriptor.__get__))  # <class 'function'>
print(type(MyDescriptor.__get__.__get__))  # <class 'method-wrapper'>
print(type(MyDescriptor.__get__.__init__))  # <class 'method-wrapper'>
print(type(MyDescriptor.__init__)) # <class 'function'>
print(type(MyDescriptor.__init__.__get__)) # <class 'method-wrapper'>
print(type(user_func)) # <class 'function'>
print(type(user_func.__get__))  # <class 'method-wrapper'>

# user_func 虽然是 function 类型（描述符），但 user 实例没有 user_func 属性
# 描述符协议只在属性访问时生效，需要将 user_func 作为类属性定义
# print(user.user_func)  # 这会报错：AttributeError: 'MyDescriptor' object has no attribute 'user_func'

# 正确的做法是将 user_func 作为类属性：
class MyClass:
    user_func = user_func  # 将函数作为类属性

obj = MyClass()
print(obj.user_func)  # 这样可以正常工作，因为 user_func 作为描述符被绑定到实例

