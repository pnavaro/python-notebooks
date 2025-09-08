alphabet = "abcdefghijklmnopqrstuvwxyz"


def cipher(s, shift=0):
    crypted = s
    for c in s:
        k = alphabet.index(c)
        crypted = crypted.replace(c, alphabet[(k + shift) % 26], 1)

    return crypted


def plain(s, shift=0):
    return cipher(s, -shift)


cipher("caesar", 45)
