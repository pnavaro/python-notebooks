def ip_to_int(a, b, c, d):
    return (a << 24) + (b << 16) + (c << 8) + d

def mask(ip1, ip2):
    "ip1 and ip2 are lists of 4 integers 0-255 each"
    m = 0xFFFFFFFF ^ ip_to_int(*ip1) ^ ip_to_int(*ip2)
    return [(m & (0xFF << (8*n))) >> 8*n for n in (3, 2, 1, 0)]

print(mask([192, 168, 1, 1], [192, 168, 1, 254]))
