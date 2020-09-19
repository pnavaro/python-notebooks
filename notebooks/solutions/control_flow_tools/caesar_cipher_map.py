def cipher(text, key):
    l_alphabet = "abcdefghijklmnopqrstuvwxyz"
    u_alphabet = l_alphabet.upper()

    def shift(c):
        if c.islower():
            return l_alphabet[(l_alphabet.index(c) + key) % 26]
        else:
            return u_alphabet[(u_alphabet.index(c) + key) % 26]

    return map(shift, text)


crypted_text = cipher("Python", 5)
print(*crypted_text)


def plain(text, key):
    return cipher(text, -key)


print(*plain(cipher("Python", 5), 5))
