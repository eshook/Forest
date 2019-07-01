
# define PyCUDA imports
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
from pycuda.tools import DeviceData
from pycuda.compiler import SourceModule
from pycuda.characterize import sizeof
import numpy as np

# create kernel function
# defined as a multiline python string. Written in C++
code = '''
	__global__ void doublify(float *data) {

		int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int thread_id = y * grid_size + x;
		data[thread_id] *= 2;
	}
'''

# compile kernel code
mod = SourceModule(kernel_code, no_extern_c = True)

# get kernel function
func = mod.get_function('doublify')

# create data
data1 = np.random.randn(4,4).astype(np.float32) # must be np.float32 or np.int32 to run on GPU
data1 = gpuarray.to_gpu(data1)
data2 = np.random.randn(4,4).astype(np.float32) # must be np.float32 or np.int32 to run on GPU
data2 = gpuarray.to_gpu(data2)

# WITH FOREST

# define a primitive in PrimitivesRaster.py file
class Hello_kernel_func(Primitive):
    def __call__(self):
    	self.action(self.data, grid = (self.grid_dims, self.grid_dims), block = (self.block_dims, self.block_dims, 1))
    def vars(self,func,data,grid_block):
        self.action = func
        self.data = data
        self.grid_dims = grid
        self.block_dims = block
        return self # Must still return self so there is something to call
hello_kernel_func = Hello_kernel_func()

# pass inputs into function using vars()
# call function inside run_primitive()
# ==, <=, >=, etc. separate different kernel calls
run_primitive(hello_kernel_func.vars(func, data1, 1, 4) == hello_kernel_func.vars(func, data2, 1, 4))

# WITHOUT FOREST

# call function with data, other parameters, grid size, and block size
func(data, grid = (1, 1), block = (4, 4))

# A PBS script to submit to the GPU will also need to be created
# Examples can be found here: https://github.com/tyburesh

