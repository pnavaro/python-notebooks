s = "python LILLE 2018"
res = s.split(" ")
print(res)
res = res[:2] + ["april", 10] + res[2:]
print(res)
res[0] = res[0].capitalize()
print(res)
d = dict(course=res[0], month=res[2], day=res[3], year=res[4])
print(d.keys)
print(d.items())
d["place"] = res[1]
print(d)
