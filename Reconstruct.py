import os

print('141 x 2 ================> 281 Start!')
os.system('CUDA_VISIBLE_DEVICES=1 python test_141.py')
print('141 x 2 ================> 281 Complete!')
print('281 x 2.5 ==============> 701 Start!')
os.system('CUDA_VISIBLE_DEVICES=1 python test_281.py')
print('281 x 2.5 ==============> 701 Complete!')
print('141 =======> 281 =======> 701')
os.system('CUDA_VISIBLE_DEVICES=1 python show_deepsd_demo.py')
