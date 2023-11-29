import time


rate=10000
while True:
    now=time.time()
    while time.time()-now<1/rate:
        continue
    print(time.time()-now)