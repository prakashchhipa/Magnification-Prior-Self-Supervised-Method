'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

image_size = 224
image_mean = [0.4914, 0.4822, 0.4465]
image_std = [0.2023, 0.1994, 0.2010]
batch_size = 32#1 #for test, otherwise 7
num_workers = 1

gpu0 = 'cuda:0'
gpu1 = 'cuda:1'
gpu2 = 'cuda:2'
gpu3 = 'cuda:3'
gpu4 = 'cuda:4'
gpu5 = 'cuda:5'
gpu6 = 'cuda:6'

result_path = '/home/result/results_bc/'
tensorboard_path = '/home/logs/tensorboard_bc/'
