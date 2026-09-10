#Example 2.9
def maxarray (xs):
 m = xs[0]
 for x in xs:
  if m < x:
   m = x
 return m

def sortarray (xs):
 for i in range(len(xs)):
  for j in range(i + 1, len(xs)):
   if xs[i] > xs[j]:
    xs[i], xs[j] = xs[j], xs[i]
 return xs

data = [3,1,4,0,2,5]
t = maxarray(data)
print(t)
print(sortarray(data))