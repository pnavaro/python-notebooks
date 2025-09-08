def is_happy(n):
    """ Check if a number is happy """
    unhappy_list = []
    while True:
        r = 0
        for d in str(n):
            r += int(d) * int(d)
        if r == 1:
            return True
        elif r in unhappy_list:  # If a member of its sequence is unhappy
            return False  # then the number is unhappy
        unhappy_list.append(r)
        n = r


def happy(n):
    """Return happy numbers < n"""
    return [i for i in range(1, n) if is_happy(i)]


print(happy(100))
