@property
def standalone_function(self, x):
    return f"处理 {x} 来自 {self}"

print({type(standalone_function.__get__)})