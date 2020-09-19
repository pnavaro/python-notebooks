def reverse(n):
    return int(str(n)[::-1])


def skinny(n):
    """
    Non palindromic skinny numbers
    :param n:
    :return: a list with n first numbers
    """
    res = []
    for m in range(10, n):
        if reverse(m ** 2) == reverse(m) ** 2 and m != reverse(m):
            res.append(m)
    return res


if __name__ == "__main__":
    print(skinny(200))
