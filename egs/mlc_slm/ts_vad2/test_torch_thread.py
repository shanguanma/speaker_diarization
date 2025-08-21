import torch
import time
# print(f'{torch.get_num_threads()}') # default is 256, it isn't important how many we set, but the fact that we set num_threads is critical
#torch.set_num_threads(32)
#torch.set_num_interop_threads(32)
x = torch.zeros(13456, 4) # starts at some size of first dim
z_out = torch.zeros(1024, 4, dtype=torch.uint8)

start = time.time()

for _ in range(64):
    (x == x).all() 

print(time.time() - start)
