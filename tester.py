from random import sample
import os
path_dir = 'image/sample'
file_list = os.listdir(path_dir)
return sample(file_list, 21)