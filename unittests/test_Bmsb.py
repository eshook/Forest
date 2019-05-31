
from test_forest_bmsb_imports import *
from forest import *
import unittest
import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom

class TestBmsb(unittest.TestCase):

	def setUp(self):
		self.engine = cuda_engine
		self.engine.stack = Stack()
		self.engine.queue = Queue()
		self.engine.is_split = False
		self.engine.continue_cycle = False
		self.engine.n_iters = 0
		self.engine.iters = 0

	def test_cuda_engine(self):

		self.assertFalse(self.engine.is_split)
		self.assertFalse(self.engine.continue_cycle)
		self.assertEqual(self.engine.n_iters, 0)
		self.assertEqual(self.engine.iters, 0)

	def test_grid_initialization(self):

		prim = Initialize_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		grid_a = self.engine.stack.pop().data
		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		grid_b[MATRIX_SIZE//2][MATRIX_SIZE//2] = 1
		self.assertTrue((grid_a == grid_b).all())

		prim = Empty_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		grid_c = self.engine.stack.pop().data
		grid_d = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		self.assertTrue((grid_c == grid_d).all())

	def test_split_single_grid(self):
		
		prim = Empty_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		self.engine.split()
		grid_gpu = self.engine.stack.pop()
		self.assertTrue(self.engine.is_split)
		self.assertIsInstance(grid_gpu, gpuarray.GPUArray)

	def test_split_multiple_grid(self):

		num_grids = 2
		for i in range(num_grids):
			prim = Empty_grid()
			run_primitive(prim.size(MATRIX_SIZE))
		self.engine.split()
		self.assertTrue(self.engine.is_split)
		for i in range(num_grids):
			grid_gpu = self.engine.stack.pop()
			self.assertIsInstance(grid_gpu, gpuarray.GPUArray)

	def test_merge_single_grid(self):
		
		# make sure merge works for a single grid
		grid = Raster(h=MATRIX_SIZE,w=MATRIX_SIZE,nrows=MATRIX_SIZE,ncols=MATRIX_SIZE)
		grid.data = grid.data.astype(np.float32)
		grid_gpu = gpuarray.to_gpu(grid.data)
		self.engine.stack.push(grid_gpu)
		self.engine.merge()
		grid_cpu = self.engine.stack.pop()
		self.assertFalse(self.engine.is_split)
		self.assertIsInstance(grid_cpu, np.ndarray)

	def test_merge_multiple_grids(self):

		# make sure merge works for two or more grids
		num_grids = 2
		for i in range(num_grids):
			grid = Raster(h=MATRIX_SIZE,w=MATRIX_SIZE,nrows=MATRIX_SIZE,ncols=MATRIX_SIZE)
			grid.data = grid.data.astype(np.float32)
			grid_gpu = gpuarray.to_gpu(grid.data)
			self.engine.stack.push(grid_gpu)
		self.engine.merge()
		self.assertFalse(self.engine.is_split)
		for i in range(num_grids):
			grid_cpu = self.engine.stack.pop()
			self.assertIsInstance(grid_cpu, np.ndarray)

	def test_cycle_start(self):
		
		prim = Empty_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		prim = Initialize_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		self.engine.cycle_start()

		self.assertTrue(self.engine.is_split)
		self.assertTrue(self.engine.continue_cycle)

	# Cycle termination loops inifinitely for some reason right now
	def test_cycle_termination(self):
		
		prim = Empty_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		prim = Initialize_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		self.engine.n_iters = N_ITERS
		self.engine.cycle_start()
		prim = Local_diffusion()
		run_primitive(prim.vars(LOCAL_ALWAYS, GRID_DIMS, BLOCK_DIMS))
		prim = Bmsb_stop()
		run_primitive(prim)
		self.engine.cycle_termination()

		self.assertEqual(self.engine.iters, N_ITERS)
		self.assertFalse(self.engine.is_split)
		self.assertFalse(self.engine.continue_cycle)

	def test_bmsb_stop_condition(self):

		prim = Bmsb_stop_condition()
		run_primitive(prim.vars(N_ITERS))
		self.assertEqual(self.engine.n_iters, N_ITERS)

	def test_bmsb_stop(self):

		self.engine.continue_cycle = True
		for i in range(N_ITERS):
			prim = Bmsb_stop()
			run_primitive(prim)
		self.assertEqual(self.engine.iters, N_ITERS)
		self.assertFalse(self.engine.continue_cycle)

	def test_local_diffusion_always(self):

		prim = Empty_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		prim = Initialize_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		self.engine.split()
		prim = Local_diffusion()
		run_primitive(prim.vars(LOCAL_ALWAYS, GRID_DIMS, BLOCK_DIMS))
		grid_a = self.engine.stack.pop().get()
		
		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		for i in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
			for j in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
				grid_b[i][j] = 1

		self.assertTrue((grid_a == grid_b).all())

	def test_local_diffusion_never(self):

		prim = Empty_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		prim = Initialize_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		self.engine.split()
		prim = Local_diffusion()
		run_primitive(prim.vars(LOCAL_NEVER, GRID_DIMS, BLOCK_DIMS))
		grid_a = self.engine.stack.pop().get()

		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		grid_b[MATRIX_SIZE//2][MATRIX_SIZE//2] = 1

		self.assertTrue((grid_a == grid_b).all())

	def test_local_diffusion_edge(self):

		prim = Empty_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		prim = Empty_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		bob = self.engine.stack.pop()
		bob.data[0][0] = 1
		self.engine.stack.push(bob)
		self.engine.split()
		prim = Local_diffusion()
		run_primitive(prim.vars(LOCAL_ALWAYS, GRID_DIMS, BLOCK_DIMS))
		grid_a = self.engine.stack.pop().get()

		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		grid_b[0][0] = 1
		self.assertTrue((grid_a == grid_b).all())

	def test_non_local_diffusion_once(self):

		prim = Empty_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		prim = Initialize_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		self.engine.split()
		prim = Non_local_diffusion()
		run_primitive(prim.vars(NON_LOCAL_ONCE, GRID_DIMS, BLOCK_DIMS))
		grid_a = self.engine.stack.pop().get()

		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		grid_b[MATRIX_SIZE//2][MATRIX_SIZE//2] = 1
		grid_b[MATRIX_SIZE-1][MATRIX_SIZE-1] = 1
		print('Grid_a = ', grid_a)
		print('Grid_b = ', grid_b)
		self.assertTrue((grid_a == grid_b).all())

	def test_non_local_diffusion_never(self):
		
		prim = Empty_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		prim = Initialize_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		self.engine.split()
		prim = Non_local_diffusion()
		run_primitive(prim.vars(NON_LOCAL_NEVER, GRID_DIMS, BLOCK_DIMS))
		grid_a = self.engine.stack.pop().get()

		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		grid_b[MATRIX_SIZE//2][MATRIX_SIZE//2] = 1
		print('Grid_a = ', grid_a)
		print('Grid_b = ', grid_b)
		self.assertTrue((grid_a == grid_b).all())

	def test_survival_kernel_none_survive(self):
		
		# make sure all cells die
		prim = Empty_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		prim = Empty_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		bob = self.engine.stack.pop()
		for i in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
			for j in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
				bob.data[i][j] = 1
		self.engine.stack.push(bob)
		self.engine.split()
		prim = Survival_of_the_fittest()
		run_primitive(prim.vars(SURVIVAL_NONE, GRID_DIMS, BLOCK_DIMS))
		grid_a = self.engine.stack.pop().get()

		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		self.assertTrue((grid_a == grid_b).all())

	# Need to figure out how to change P_DEATH in play_bmsb before running test
	def test_survival_kernel_all_survive(self):
		
		# make sure no cells die
		prim = Empty_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		prim = Empty_grid()
		run_primitive(prim.size(MATRIX_SIZE))
		bob = self.engine.stack.pop()
		for i in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
			for j in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
				bob.data[i][j] = 1
		self.engine.stack.push(bob)
		self.engine.split()
		prim = Survival_of_the_fittest()
		run_primitive(prim.vars(SURVIVAL_ALL, GRID_DIMS, BLOCK_DIMS))
		grid_a = self.engine.stack.pop().get()

		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		for i in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
			for j in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
				grid_b[i][j] = 1
		self.assertTrue((grid_a == grid_b).all())

	# Would be better to figure out how to set the same seed every time
	def test_get_random_number(self):

		grid = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		grid = gpuarray.to_gpu(grid)
		self.engine.generator = curandom.XORWOWRandomNumberGenerator(curandom.seed_getter_uniform)
		for i in range(10):
			RAND_NUM(self.engine.generator.state, grid, grid = (GRID_DIMS,GRID_DIMS,1), block = (BLOCK_DIMS,BLOCK_DIMS,1))
			grid_cpu = grid.get()
			for i in range(MATRIX_SIZE):
				for j in range(MATRIX_SIZE):
					self.assertGreaterEqual(grid_cpu[i][j], 0)
					self.assertLess(grid_cpu[i][j], 1)

	# Would be better to figure out how to set the same seed every time
	def test_get_random_cell(self):
		
		grid = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		grid = gpuarray.to_gpu(grid)
		self.engine.generator = curandom.XORWOWRandomNumberGenerator(curandom.seed_getter_uniform)
		for i in range(10):
			RAND_CELL(self.engine.generator.state, grid, grid = (GRID_DIMS,GRID_DIMS,1), block = (BLOCK_DIMS,BLOCK_DIMS,1))
			grid_cpu = grid.get()
			for i in range(MATRIX_SIZE):
				for j in range(MATRIX_SIZE):
					self.assertGreaterEqual(grid_cpu[i][j], 0)
					self.assertLess(grid_cpu[i][j], MATRIX_SIZE * MATRIX_SIZE)

	def test_pop2data2gpu(self):

		grid_a = np.zeros((MATRIX_SIZE,MATRIX_SIZE))
		grid_b = np.ones((MATRIX_SIZE,MATRIX_SIZE))
		self.engine.stack.push(grid_a)
		self.engine.stack.push(grid_b)

		@pop2data2gpu
		def func(a,b):
			np.add(a,b)
			return a,b

		grid_c = np.ones((MATRIX_SIZE,MATRIX_SIZE))
		self.assertTrue((self.engine.stack.pop() == grid_c).all())


# Create the TestBmsb suite        
test_Bmsb_suite = unittest.TestLoader().loadTestsFromTestCase(TestBmsb)
