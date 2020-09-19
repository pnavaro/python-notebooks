l_alphabet = "abcdefghijklmnopqrstuvwxyz"
u_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def cipher(text, shift):
    crypted_text = ""
    for c in text:
        for i, (lo, up) in enumerate(zip(l_alphabet, u_alphabet)):
            if lo == c:
                crypted_text += l_alphabet[(i + shift) % 26]
            elif up == c:
                crypted_text += u_alphabet[(i + shift) % 26]

    return crypted_text


def plain(text, shift):
    return cipher(text, -shift)


s = cipher("Python", 13)
print(s)
print(plain(s, 13))
