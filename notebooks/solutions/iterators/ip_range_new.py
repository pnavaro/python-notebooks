def ip_to_int(a, b, c, d):
    return (a << 24) + (b << 16) + (c << 8) + d

def int_to_ip( ip ):
    d = ip
    a = ip >> 24
    d = d - (a << 24)
    b = d >> 16
    d = d - (b << 16)
    c = d >> 8
    d = d - (c << 8)

    return a, b, c, d

def ip_range_new(start_ip, end_ip):
   start = list(map(int, start_ip.split(".")))
   end = list(map(int, end_ip.split(".")))

   for ip in range(ip_to_int(*start), ip_to_int(*end)+1):
      yield ".".join(map(str,int_to_ip(ip)))
        
for ip in ip_range_new("192.168.1.0", "192.168.1.10"):
   print(ip)

