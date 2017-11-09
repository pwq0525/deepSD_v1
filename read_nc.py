'''数据文件目录为dem（地形），input（５０公里降水），label（１公里降水）

util_data里面需要调用的函数就是input_setup,传入sess和config,
    输出arr_input的形状是(32, 24, 15, 15),＃文件数×２４个时况×１５×１５
    arr_dem的形状是(1, 701, 701),max_value:2944,min_value:-9
    arr_label的形状是(32, 24, 700, 700),
    其他为内置函数，不用管
'''
import netCDF4 as nc
import tensorflow as tf
import os
import glob
import scipy.ndimage
import numpy as np


'''获取文件夹里，nc文件的路径列表'''
def prepare_data(dataset):

    #获取ｎｃ文件路径列表
    filenames = os.listdir(dataset)
    # os.getcwd()：获取当前工作目录，也就是在哪个目录下运行这个程序。
    data_dir = os.path.join(os.getcwd(), dataset)
    # glob函数用来查找符合指定规则的文件路径名
    data = glob.glob(os.path.join(data_dir, "*.nc"))

    return data
'''
def main():
	d = get_data('dem')
	print("dem:",d.shape)
	d = get_data('input')
	print("input:",d.shape)
	d = get_data('label')
	print("label:",d.shape)
'''

def get_dimen(config):
	paths = prepare_data(config.label)
	path = paths[0]
	d = []
	dataset = nc.Dataset(path)
	keys = list(dataset.variables.keys())
	d.append(dataset.variables[keys[0]][:])
	for key in keys[1:-1]:
		data = dataset.variables[key][:]
		d.append(scipy.ndimage.zoom(data,(config.label_size/len(data))))
	dataset.close()
	return d

def get_name(name,label_size=''):
	paths = prepare_data(name)
	paths = sorted(paths)
	names = [path.split('/')[-1] for path in paths]
	return [str('output_'+label_size+'_'+name.split('_')[-1]) for name in names]

def get_data(name):
	paths = prepare_data(name)
	paths = sorted(paths)
	#print(paths)
	d = []
	for path in paths:
		#print(path)
		d.append(progress(path))
	d = np.array(d)
	d = d.reshape(-1,d.shape[-2],d.shape[-1])
	return d

'''处理每个路径下的文件，得到每个文件的降水或地形数据,其中地形和降水标签需要缩放'''
def progress(path,scale=2):
    #获取数据
    type = path.split('/')[-2]
    #print(type)
    data = nc.Dataset(path)

    '''分类型处理'''
    #地形
    if type == "dem":#3541* 6165　todo 这里降水和地形的数据都是:lat:43-36,lon:113-120了
        d =data.variables['dem'][:]
        #print(data.variables['lat'][0])
        #print(data.variables['lon'][0])
        #print('dem: ',data.variables['dem'].units)
    #降水输入
    elif "input_" in type:#24*15*15
        #print('input: ',data.variables['precipitation'].units)
        d = data.variables['precipitation'][:]###################################################

    #降水标签
    elif "label_" in type:#24*701*701
        #print('label: ',data.variables['precipitation'].units)
        #d = data.variables['precipitation'][:]####################################################
        d = data.variables['precipitation'][:]####################################################

    #降水标签
    elif "output_" in type:#24*701*701
        #print('label: ',data.variables['precipitation'].units)
        #d = data.variables['precipitation'][:]####################################################
        d = data.variables['precipitation'][:]####################################################


    #降水标签
    elif "_set" in type:#24*701*701
        #print('label: ',data.variables['precipitation'].units)
        d = data.variables['precipitation'][:]####################################################

    '''
    #降水输入
    elif type =="input_15":#24*15*15
        #print(data.variables.keys())
        d = data.variables['precipitation'][:]
        #print(data.variables['lat'][0])
        #print(data.variables['lon'][0])
    #降水标签
    elif type == "label_15":#24*701*701
        #print(data.variables.keys())
        d = data.variables['precipitation'][:]
        #print(data.variables['lat'][0])
        #print(data.variables['lon'][0])
    '''

    data.close()

    return d
'''
if __name__=='__main__':
  main()
'''
