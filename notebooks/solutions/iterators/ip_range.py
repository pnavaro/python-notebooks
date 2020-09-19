def ip_range(start_ip, end_ip):
   start = list(map(int, start_ip.split(".")))
   end = list(map(int, end_ip.split(".")))
   ip_addr = start # tuple with 3 elements 
   
   ip = start_ip
   while ip_addr != end:
      start[3] += 1
      for i in (3, 2, 1):
         if ip_addr[i] == 256:
            ip_addr[i] = 0
            ip_addr[i-1] += 1
      ip = ".".join(map(str, ip_addr))
      yield ip
      
   
for ip in ip_range("192.168.1.0", "192.168.1.10"):
   print(ip) 

