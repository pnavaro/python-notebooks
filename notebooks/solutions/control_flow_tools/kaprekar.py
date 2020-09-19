def kaprekar(n):
    s = str(n*n)
    m = len(s)//2
    if n == 1: return True
    if m < 1: return False
    if int(s[:m]) + int(s[m:]) == n:
        return True
    else:
        return False
    
print(*filter(kaprekar,range(10000)))
