def cipher(text, key):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    ll = [alphabet[(alphabet.index(c) + key) % 26] if c.islower() else c for c in text]
    alphabet = alphabet.upper()
    u = [alphabet[(alphabet.index(c) + key) % 26] if c.isupper() else c for c in ll]
    return ''.join(u)


def plain(text, key):
    return cipher(text, -key)


print(cipher("Python", 5))
s = cipher("Python", 5)
plain(s, 5)
