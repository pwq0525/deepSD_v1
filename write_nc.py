import netCDF4 as nc
import numpy as np

def writenc(name,data):
	dataset = nc.Dataset(name,'w',format='NETCDF4_CLASSIC')
	keys = ['time','lat','lon','precipitation']
	for i in range(len(keys)-1):
		dataset.createDimension(keys[i], len(data[i]))
		dataset.createVariable(keys[i], np.float64, (keys[i]))
		dataset.variables[keys[i]][:] = data[i]
	dataset.createVariable(keys[-1], np.float32, tuple(keys[:-1]))
	data[-1] = np.maximum(data[-1],0)
	dataset.variables[keys[-1]][:] = data[-1]
	dataset.close()


'''
if __name__=='__main__':
	main()
'''
