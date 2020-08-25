def HistogramArea(arr):

  # code goes here
  n = len(arr)
  arr.append(0)
  stack = [-1]
  max_area = 0
  for i in range(n+1):
    while arr[i]<arr[stack[-1]]:
      h = arr[stack.pop()]
      w = i - stack[-1] -1
      max_area = max(max_area,h*w)
    stack.append(i)
  return max_area

# keep this function call here 
print(HistogramArea(input()))
