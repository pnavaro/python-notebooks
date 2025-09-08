def longest_subseq(array):
    length = [1] * len(array)
    for i,a in enumerate(array[:]):
    
        if a-1 in array[:i]:
            length[i] = length[ array.index(a-1) ] + 1
        else:
            length[i] = 1
            
    return max(length)

print(longest_subseq([3, 10, 3, 11, 4, 5, 6, 7, 8, 12]))
print(longest_subseq([6, 7, 8, 3, 4, 5, 9, 10]))