alphabet = "abcdefghijklmnopqrstuvwxyz"


def cipher(text, key):
    """ Crypt text using Caesar cipher"""

    crypted_text = ""
    for c in text:
        for i, l in enumerate(alphabet):
            if c == l:
                crypted_text += alphabet[(i + key) % 26]

    return crypted_text


def plain(text, key):
    """ Uncrypt text using Caesar cipher"""
    return cipher(text, -key)


s = cipher("python", 13)
print(s)
plain(s, 13)
