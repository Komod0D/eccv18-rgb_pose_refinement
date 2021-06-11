import time
from utils.train_test import *
import torch

for sample in generate_samples():
    images, renders, ds, vs, poses_src, poses_tgt = sample
    print(images)
    exit(0)

network = load_network()
params = [param for _, param in network.tune_parameters()]
optimizer = torch.optim.Adam(params, 3 * 10e-4)

max_epochs = 25
for epoch in range(max_epochs):
    start = time.time()
    avg_loss = train(network, optimizer)
    print(f'------------------------------------- EPOCH {epoch + 1} FINISHED ----------------------------------------')
    print(f'epoch [{epoch + 1}/{max_epochs}], avg loss {avg_loss}, in time {time.time() - start}')
    print(f'---------------------------------------------------------------------------------------------------------')

