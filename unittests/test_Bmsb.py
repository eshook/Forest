
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
		Config.engine = CUDAEngine()

	def test_stack(self):
		stack = [1,2]
		Config.engine.stack.push(1)
		Config.engine.stack.push(2)
		self.assertTrue(Config.engine.stack.notempty())
		self.assertTrue(Config.engine.stack.size() == len(stack))
		self.assertTrue(Config.engine.stack.pop() == stack.pop())
		self.assertTrue(Config.engine.stack.pop() == stack.pop())
		self.assertFalse(Config.engine.stack.notempty())

	def test_queue(self):
		queue = [1,2]
		Config.engine.queue.enqueue(2)
		Config.engine.queue.enqueue(1)
		self.assertTrue(Config.engine.queue.notempty())
		self.assertTrue(Config.engine.queue.size() == len(queue))
		self.assertTrue(Config.engine.queue.dequeue() == queue.pop())
		self.assertTrue(Config.engine.queue.dequeue() == queue.pop())
		self.assertFalse(Config.engine.queue.notempty())

	def test_cuda_engine(self):

		self.assertFalse(Config.engine.is_split)
		self.assertFalse(Config.engine.continue_cycle)
		self.assertEqual(Config.engine.n_iters, 0)
		self.assertEqual(Config.engine.iters, 0)
		self.assertIsInstance(Config.engine.generator, curandom.XORWOWRandomNumberGenerator)
		#self.assertIsInstance(Config.engine.stack, ..forest.engines.Engine.Stack)
		#self.assertIsInstance(Config.engine.bob_stack, ..forest.engines.Engine.Stack)
		#self.assertIsInstance(Config.engine.queue, ..forest.engines.Engine.Queue)

	def test_initialize_grid(self):

		run_primitive(Initialize_grid().size(MATRIX_SIZE))
		grid_a = Config.engine.stack.pop().data
		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		grid_b[MATRIX_SIZE//2][MATRIX_SIZE//2] = 1
		self.assertTrue((grid_a == grid_b).all())

	def test_empty_grid(self):

		run_primitive(Empty_grid().size(MATRIX_SIZE))
		grid_a = Config.engine.stack.pop().data
		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		self.assertTrue((grid_a == grid_b).all())

	def test_split_single_grid(self):
		
		run_primitive(Empty_grid().size(MATRIX_SIZE))
		Config.engine.split()
		grid_gpu = Config.engine.stack.pop()
		self.assertTrue(Config.engine.is_split)
		self.assertIsInstance(grid_gpu, gpuarray.GPUArray)

	def test_split_multiple_grids(self):

		run_primitive(Empty_grid().size(MATRIX_SIZE) == Empty_grid().size(MATRIX_SIZE))
		Config.engine.split()
		self.assertTrue(Config.engine.is_split)
		while Config.engine.stack.notempty():
			grid_gpu = Config.engine.stack.pop()
			self.assertIsInstance(grid_gpu, gpuarray.GPUArray)

	def test_merge_single_grid(self):
		
		grid = Raster(h=MATRIX_SIZE,w=MATRIX_SIZE,nrows=MATRIX_SIZE,ncols=MATRIX_SIZE)
		grid.data = grid.data.astype(np.float32)
		grid_gpu = gpuarray.to_gpu(grid.data)
		Config.engine.stack.push(grid_gpu)
		Config.engine.merge()
		grid_cpu = Config.engine.stack.pop()
		self.assertFalse(Config.engine.is_split)
		self.assertIsInstance(grid_cpu, np.ndarray)

	def test_merge_multiple_grids(self):

		num_grids = 2
		for i in range(num_grids):
			grid = Raster(h=MATRIX_SIZE,w=MATRIX_SIZE,nrows=MATRIX_SIZE,ncols=MATRIX_SIZE)
			grid.data = grid.data.astype(np.float32)
			grid_gpu = gpuarray.to_gpu(grid.data)
			Config.engine.stack.push(grid_gpu)
		Config.engine.merge()
		self.assertFalse(Config.engine.is_split)
		for i in range(num_grids):
			grid_cpu = Config.engine.stack.pop()
			self.assertIsInstance(grid_cpu, np.ndarray)

	def test_cycle_start(self):

		run_primitive(Empty_grid().size(MATRIX_SIZE) == Initialize_grid().size(MATRIX_SIZE))
		Config.engine.cycle_start()

		self.assertTrue(Config.engine.is_split)
		self.assertTrue(Config.engine.continue_cycle)

	def test_cycle_termination(self):

		run_primitive(Empty_grid().size(MATRIX_SIZE) == Initialize_grid().size(MATRIX_SIZE))
		Config.engine.n_iters = N_ITERS
		Config.engine.cycle_start()
		run_primitive(Local_diffusion().vars(LOCAL_ALWAYS, GRID_DIMS, BLOCK_DIMS) == Bmsb_stop())
		Config.engine.cycle_termination()

		self.assertEqual(Config.engine.iters, N_ITERS)
		self.assertFalse(Config.engine.is_split)
		self.assertFalse(Config.engine.continue_cycle)

	def test_bmsb_stop_condition(self):

		run_primitive(Bmsb_stop_condition().vars(N_ITERS))
		self.assertEqual(Config.engine.n_iters, N_ITERS)

	def test_bmsb_stop(self):

		Config.engine.continue_cycle = True
		for i in range(N_ITERS):
			run_primitive(Bmsb_stop())
		self.assertEqual(Config.engine.iters, N_ITERS)
		self.assertFalse(Config.engine.continue_cycle)

	def test_local_diffusion_always(self):

		run_primitive(Empty_grid().size(MATRIX_SIZE) == Initialize_grid().size(MATRIX_SIZE))
		Config.engine.split()
		run_primitive(Local_diffusion().vars(LOCAL_ALWAYS, GRID_DIMS, BLOCK_DIMS))
		grid_a = Config.engine.stack.pop().get()
		
		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		for i in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
			for j in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
				grid_b[i][j] = 1

		self.assertTrue((grid_a == grid_b).all())

	def test_local_diffusion_never(self):

		run_primitive(Empty_grid().size(MATRIX_SIZE) == Initialize_grid().size(MATRIX_SIZE))
		Config.engine.split()
		run_primitive(Local_diffusion().vars(LOCAL_NEVER, GRID_DIMS, BLOCK_DIMS))
		grid_a = Config.engine.stack.pop().get()

		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		grid_b[MATRIX_SIZE//2][MATRIX_SIZE//2] = 1

		self.assertTrue((grid_a == grid_b).all())

	def test_local_diffusion_edge(self):

		run_primitive(Empty_grid().size(MATRIX_SIZE) == Empty_grid().size(MATRIX_SIZE))
		bob = Config.engine.stack.pop()
		bob.data[0][0] = 1
		Config.engine.stack.push(bob)
		Config.engine.split()
		run_primitive(Local_diffusion().vars(LOCAL_ALWAYS, GRID_DIMS, BLOCK_DIMS))
		grid_a = Config.engine.stack.pop().get()

		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		grid_b[0][0] = 1
		self.assertTrue((grid_a == grid_b).all())

	def test_non_local_diffusion_once(self):

		run_primitive(Empty_grid().size(MATRIX_SIZE) == Initialize_grid().size(MATRIX_SIZE))
		Config.engine.split()
		run_primitive(Non_local_diffusion().vars(NON_LOCAL_ONCE, GRID_DIMS, BLOCK_DIMS))
		grid_a = Config.engine.stack.pop().get()

		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		grid_b[MATRIX_SIZE//2][MATRIX_SIZE//2] = 1
		grid_b[MATRIX_SIZE-1][MATRIX_SIZE-1] = 1
		print('Grid_a = ', grid_a)
		print('Grid_b = ', grid_b)
		self.assertTrue((grid_a == grid_b).all())

	def test_non_local_diffusion_never(self):
		
		run_primitive(Empty_grid().size(MATRIX_SIZE) == Initialize_grid().size(MATRIX_SIZE))
		Config.engine.split()
		run_primitive(Non_local_diffusion().vars(NON_LOCAL_NEVER, GRID_DIMS, BLOCK_DIMS))
		grid_a = Config.engine.stack.pop().get()

		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		grid_b[MATRIX_SIZE//2][MATRIX_SIZE//2] = 1
		print('Grid_a = ', grid_a)
		print('Grid_b = ', grid_b)
		self.assertTrue((grid_a == grid_b).all())

	def test_survival_kernel_none_survive(self):
		
		# make sure all cells die
		run_primitive(Empty_grid().size(MATRIX_SIZE) == Empty_grid().size(MATRIX_SIZE))
		bob = Config.engine.stack.pop()
		for i in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
			for j in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
				bob.data[i][j] = 1
		Config.engine.stack.push(bob)
		Config.engine.split()
		run_primitive(Survival_of_the_fittest().vars(SURVIVAL_NONE, GRID_DIMS, BLOCK_DIMS))
		grid_a = Config.engine.stack.pop().get()

		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		self.assertTrue((grid_a == grid_b).all())

	# Need to figure out how to change P_DEATH in play_bmsb before running test
	def test_survival_kernel_all_survive(self):
		
		# make sure no cells die
		run_primitive(Empty_grid().size(MATRIX_SIZE) == Empty_grid().size(MATRIX_SIZE))
		bob = Config.engine.stack.pop()
		for i in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
			for j in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
				bob.data[i][j] = 1
		Config.engine.stack.push(bob)
		Config.engine.split()
		run_primitive(Survival_of_the_fittest().vars(SURVIVAL_ALL, GRID_DIMS, BLOCK_DIMS))
		grid_a = Config.engine.stack.pop().get()

		grid_b = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		for i in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
			for j in range((MATRIX_SIZE//2)-1,(MATRIX_SIZE//2)+2):
				grid_b[i][j] = 1
		self.assertTrue((grid_a == grid_b).all())

	# Would be better to figure out how to set the same seed every time
	def test_get_random_number(self):

		grid = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		grid = gpuarray.to_gpu(grid)
		Config.engine.generator = curandom.XORWOWRandomNumberGenerator(curandom.seed_getter_uniform)
		for i in range(10):
			RAND_NUM(Config.engine.generator.state, grid, grid = (GRID_DIMS,GRID_DIMS,1), block = (BLOCK_DIMS,BLOCK_DIMS,1))
			grid_cpu = grid.get()
			for i in range(MATRIX_SIZE):
				for j in range(MATRIX_SIZE):
					self.assertGreaterEqual(grid_cpu[i][j], 0)
					self.assertLess(grid_cpu[i][j], 1)

	# Would be better to figure out how to set the same seed every time
	def test_get_random_cell(self):
		
		grid = np.zeros((MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
		grid = gpuarray.to_gpu(grid)
		Config.engine.generator = curandom.XORWOWRandomNumberGenerator(curandom.seed_getter_uniform)
		for i in range(10):
			RAND_CELL(Config.engine.generator.state, grid, grid = (GRID_DIMS,GRID_DIMS,1), block = (BLOCK_DIMS,BLOCK_DIMS,1))
			grid_cpu = grid.get()
			for i in range(MATRIX_SIZE):
				for j in range(MATRIX_SIZE):
					self.assertGreaterEqual(grid_cpu[i][j], 0)
					self.assertLess(grid_cpu[i][j], MATRIX_SIZE * MATRIX_SIZE)

	def test_pop2data2gpu(self):

		grid_a = np.zeros((MATRIX_SIZE,MATRIX_SIZE))
		grid_b = np.ones((MATRIX_SIZE,MATRIX_SIZE))
		Config.engine.stack.push(grid_a)
		Config.engine.stack.push(grid_b)

		@pop2data2gpu
		def func(a,b):
			np.add(a,b)
			return a,b

		grid_c = np.ones((MATRIX_SIZE,MATRIX_SIZE))
		self.assertTrue((Config.engine.stack.pop() == grid_c).all())


# Create the TestBmsb suite        
test_Bmsb_suite = unittest.TestLoader().loadTestsFromTestCase(TestBmsb)
