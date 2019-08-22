import json

a = [[1, 2, 3], [4, 5, 6]]
b = "J'aime le caramel !"
c = 6

data = {'array': a, 'string': b, 'int': c}

my_file = open('data.txt', 'w')
my_file.write(json.dumps(data))
my_file = open('data.txt', 'r')
content = json.loads(my_file.read())
my_file.close()

print(content)
