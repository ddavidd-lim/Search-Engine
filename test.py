import sys
import time
from collections import defaultdict

import psutil as psutil

indextimer = time.time()
time.sleep(2)
indextimerend = time.time()
runtime = indextimerend - indextimer
print(f"Finished Indexing in {runtime:.2} s")

timerStart = time.time()
time.sleep(2)
timeEnd = time.time()
runtime = (timeEnd - timerStart) * 1000
print(f"Finished query in {runtime:.4f} ms")


my_list = [0] * 1000000

# get the size of the object in bytes
size_in_bytes = sys.getsizeof(my_list)
size_in_mb = size_in_bytes / (1024*1024)

print(f"The size of my_list is {size_in_mb:.2f} MB")


# ----------------------------
term_freqs = defaultdict(int)

























