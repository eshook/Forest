
from test_forest_bmsb_imports import *
from forest import *
import unittest
import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

class TestBmsb(unittest.TestCase):

	def setUp(self):
		self.engine = cuda_engine
		self.engine.stack = Stack()
		self.engine.queue = Queue()
		self.engine.continue_cycle = False
		self.engine.n_iters = 0
		self.engine.iters = 0

	def test_cuda_engine(self):

		self.assertFalse(self.engine.continue_cycle)
		self.assertEqual(self.engine.n_iters, 0)
		self.assertEqual(self.engine.iters, 0)

	def test_grid_initialization(self):

		prim = Initialize_grid()
		run_primitive(prim.size(10))
		grid_a = self.engine.stack.pop().data
		grid_b = np.zeros((10,10)).astype(np.float32)
		grid_b[5][5] = 1
		self.assertTrue((grid_a == grid_b).all())

		prim = Empty_grid()
		run_primitive(prim.size(10))
		grid_c = self.engine.stack.pop().data
		grid_d = np.zeros((10,10)).astype(np.float32)
		self.assertTrue((grid_c == grid_d).all())

	def test_split_single_grid(self):
		
		prim = Empty_grid()
		run_primitive(prim.size(10))
		self.engine.split()
		grid_gpu = self.engine.stack.pop()
		self.assertIsInstance(grid_gpu, gpuarray.GPUArray)

	def test_split_multiple_grid(self):

		num_grids = 2
		for i in range(num_grids):
			prim = Empty_grid()
			run_primitive(prim.size(10))
		self.engine.split()
		for i in range(num_grids):
			grid_gpu = self.engine.stack.pop()
			self.assertIsInstance(grid_gpu, gpuarray.GPUArray)

	def test_merge_single_grid(self):
		
		# make sure merge works for a single grid
		grid = Raster(h=10,w=10,nrows=10,ncols=10)
		grid.data = grid.data.astype(np.float32)
		grid_gpu = gpuarray.to_gpu(grid.data)
		self.engine.stack.push(grid_gpu)
		self.engine.merge()
		grid_cpu = self.engine.stack.pop()
		self.assertIsInstance(grid_cpu, np.ndarray)

	def test_merge_multiple_grids(self):

		# make sure merge works for two or more grids
		num_grids = 2
		for i in range(num_grids):
			grid = Raster(h=10,w=10,nrows=10,ncols=10)
			grid.data = grid.data.astype(np.float32)
			grid_gpu = gpuarray.to_gpu(grid.data)
			self.engine.stack.push(grid_gpu)
		self.engine.merge()
		for i in range(num_grids):
			grid_cpu = self.engine.stack.pop()
			self.assertIsInstance(grid_cpu, np.ndarray)

	def __test_cycle_start(self):
		pass

	def __test_cycle_termination(self):
		pass

	def test_bmsb_stop_condition(self):

		prim = Bmsb_stop_condition()
		run_primitive(prim.vars(3))
		self.assertEqual(self.engine.n_iters, 3)

	def test_bmsb_stop(self):

		self.engine.continue_cycle = True
		for i in range(5):
			prim = Bmsb_stop()
			run_primitive(prim)
		self.assertEqual(self.engine.iters, 5)
		self.assertFalse(self.engine.continue_cycle)

	def test_local_diffusion_always(self):

		prim = Empty_grid()
		run_primitive(prim.size(10))
		prim = Initialize_grid()
		run_primitive(prim.size(10))
		self.engine.split()
		prim = Local_diffusion()
		run_primitive(prim.vars(LOCAL, 5, 2))
		grid_a = self.engine.stack.pop().get()
		
		grid_b = np.zeros((10,10)).astype(np.float32)
		for i in range(4,7):
			for j in range(4,7):
				grid_b[i][j] = 1

		self.assertTrue((grid_a == grid_b).all())

	def test_local_diffusion_never(self):

		prim = Empty_grid()
		run_primitive(prim.size(10))
		prim = Initialize_grid()
		run_primitive(prim.size(10))
		self.engine.split()
		prim = Local_diffusion()
		run_primitive(prim.vars(LOCAL, 5, 2))
		grid_a = self.engine.stack.pop().get()

		grid_b = np.zeros((10,10)).astype(np.float32)
		grid_b[5][5] = 1

		print("Grid_a = {}\tGrid_b = {}".format(grid_a, grid_b))
		self.assertTrue((grid_a == grid_b).all())

	def test_local_diffusion_edge(self):

		prim = Empty_grid()
		run_primitive(prim.size(10))
		prim = Empty_grid()
		run_primitive(prim.size(10))
		bob = self.engine.stack.pop()
		bob.data[0][0] = 1
		self.engine.stack.push(bob)
		self.engine.split()
		prim = Local_diffusion()
		run_primitive(prim.vars(LOCAL, 5, 2))
		grid_a = self.engine.stack.pop().get()

		grid_b = np.zeros((10,10)).astype(np.float32)
		grid_b[0][0] = 1
		self.assertTrue((grid_a == grid_b).all())

	def __test_non_local_diffusion_kernel(self):
		pass

	def __test_survival_kernel(self):
		pass

	def __test_random_number_generation(self):
		pass

	def test_pop2data2gpu(self):

		grid_a = np.zeros((10,10))
		grid_b = np.ones((10,10))
		self.engine.stack.push(grid_a)
		self.engine.stack.push(grid_b)

		@pop2data2gpu
		def func(a,b):
			np.add(a,b)
			return a,b

		grid_c = np.ones((10,10))
		self.assertTrue((self.engine.stack.pop() == grid_c).all())


# Create the TestBmsb suite        
test_Bmsb_suite = unittest.TestLoader().loadTestsFromTestCase(TestBmsb)
