m = 10
b = 5
x = [1,2,3,4,5,6,7,8,9,10]
y = []

for i in range( len(x) ):
    y.append( m*x[i] + b )

print("Valores de x: ",x)
print("Valores de y: ",y)
