
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

	def test_initialize_grid(self):

		initial_population = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		initial_population[matrix_size//2][matrix_size//2] = 1
		survival_probabilities = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		run_primitive(Initialize_grid().vars(matrix_size, initial_population, survival_probabilities, None))
		grid_a = Config.engine.stack.pop().data
		grid_b = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		grid_b[matrix_size//2][matrix_size//2] = 1
		self.assertTrue((grid_a == grid_b).all())

	def test_empty_grid(self):

		run_primitive(Empty_grid().vars(matrix_size))
		grid_a = Config.engine.stack.pop().data
		grid_b = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		self.assertTrue((grid_a == grid_b).all())

	def test_split_single_grid(self):
		
		run_primitive(Empty_grid().vars(matrix_size))
		Config.engine.split()
		grid_gpu = Config.engine.stack.pop()
		self.assertTrue(Config.engine.is_split)
		self.assertIsInstance(grid_gpu, gpuarray.GPUArray)

	def test_split_multiple_grids(self):

		run_primitive(Empty_grid().vars(matrix_size) == Empty_grid().vars(matrix_size))
		Config.engine.split()
		self.assertTrue(Config.engine.is_split)
		while Config.engine.stack.notempty():
			grid_gpu = Config.engine.stack.pop()
			self.assertIsInstance(grid_gpu, gpuarray.GPUArray)

	def test_merge_single_grid(self):
		
		grid = Raster(h=matrix_size,w=matrix_size,nrows=matrix_size,ncols=matrix_size)
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
			grid = Raster(h=matrix_size,w=matrix_size,nrows=matrix_size,ncols=matrix_size)
			grid.data = grid.data.astype(np.float32)
			grid_gpu = gpuarray.to_gpu(grid.data)
			Config.engine.stack.push(grid_gpu)
		Config.engine.merge()
		self.assertFalse(Config.engine.is_split)
		for i in range(num_grids):
			grid_cpu = Config.engine.stack.pop()
			self.assertIsInstance(grid_cpu, np.ndarray)

	def test_cycle_start(self):

		initial_population = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		initial_population[matrix_size//2][matrix_size//2] = 1
		survival_probabilities = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		run_primitive(Empty_grid().vars(matrix_size) == Initialize_grid().vars(matrix_size, initial_population, survival_probabilities, None))
		Config.engine.cycle_start()

		self.assertTrue(Config.engine.is_split)
		self.assertTrue(Config.engine.continue_cycle)

	def test_cycle_termination(self):

		initial_population = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		initial_population[matrix_size//2][matrix_size//2] = 1
		survival_probabilities = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		generator = curandom.XORWOWRandomNumberGenerator()
		run_primitive(Empty_grid().vars(matrix_size) == Initialize_grid().vars(matrix_size, initial_population, survival_probabilities, generator))
		Config.engine.n_iters = n_iters
		Config.engine.cycle_start()
		run_primitive(Local_diffusion().vars(local_always, matrix_size, p_local_always, grid_dims, block_dims) == Bmsb_stop())
		Config.engine.cycle_termination()

		self.assertEqual(Config.engine.iters, n_iters)
		self.assertFalse(Config.engine.is_split)
		self.assertFalse(Config.engine.continue_cycle)

	def test_bmsb_stop_condition(self):

		run_primitive(Bmsb_stop_condition().vars(n_iters))
		self.assertEqual(Config.engine.n_iters, n_iters)

	def test_bmsb_stop(self):

		Config.engine.continue_cycle = True
		for i in range(n_iters):
			run_primitive(Bmsb_stop())
		self.assertEqual(Config.engine.iters, n_iters)
		self.assertFalse(Config.engine.continue_cycle)

	def test_local_diffusion_once(self):

		initial_population = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		initial_population[matrix_size//2][matrix_size//2] = 1
		survival_probabilities = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		generator = curandom.XORWOWRandomNumberGenerator()
		run_primitive(Empty_grid().vars(matrix_size) == Initialize_grid().vars(matrix_size, initial_population, survival_probabilities, generator))
		Config.engine.split()
		run_primitive(Local_diffusion().vars(local_always, matrix_size, p_local_always, grid_dims, block_dims))
		grid_a = Config.engine.stack.pop().get()
		
		grid_b = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		grid_b[(matrix_size//2)+1][matrix_size//2] = 1


		print('Local_diffusion_once\nGrid_a = {}\nGrid_b = {}'.format(grid_a, grid_b))
		self.assertTrue((grid_a == grid_b).all())

	def test_local_diffusion_twice(self):

		initial_population = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		initial_population[matrix_size//2][matrix_size//2] = 2
		survival_probabilities = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		generator = curandom.XORWOWRandomNumberGenerator()
		run_primitive(Empty_grid().vars(matrix_size) == Initialize_grid().vars(matrix_size, initial_population, survival_probabilities, generator))
		Config.engine.split()
		run_primitive(Local_diffusion().vars(local_always, matrix_size, p_local_always, grid_dims, block_dims))
		grid_a = Config.engine.stack.pop().get()
		
		grid_b = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		grid_b[(matrix_size//2)+1][matrix_size//2] = 2

		print('Local_diffusion_twice\nGrid_a = {}\nGrid_b = {}'.format(grid_a, grid_b))
		self.assertTrue((grid_a == grid_b).all())

	def test_local_diffusion_never(self):

		initial_population = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		initial_population[matrix_size//2][matrix_size//2] = 1
		survival_probabilities = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		generator = curandom.XORWOWRandomNumberGenerator()
		run_primitive(Empty_grid().vars(matrix_size) == Initialize_grid().vars(matrix_size, initial_population, survival_probabilities, generator))
		Config.engine.split()
		run_primitive(Local_diffusion().vars(local_never, matrix_size, p_local_never, grid_dims, block_dims))
		grid_a = Config.engine.stack.pop().get()

		grid_b = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		grid_b[matrix_size//2][matrix_size//2] = 1

		print('Local_diffusion_never\nGrid_a = {}\nGrid_b = {}'.format(grid_a, grid_b))
		self.assertTrue((grid_a == grid_b).all())

	def test_local_diffusion_edge(self):

		initial_population = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		initial_population[0][0] = 1
		survival_probabilities = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		generator = curandom.XORWOWRandomNumberGenerator()
		run_primitive(Empty_grid().vars(matrix_size) == Initialize_grid().vars(matrix_size, initial_population, survival_probabilities, generator))
		Config.engine.split()
		run_primitive(Local_diffusion().vars(local_always, matrix_size, p_local_always, grid_dims, block_dims))
		grid_a = Config.engine.stack.pop().get()

		grid_b = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		grid_b[0][0] = 1

		print('Local_diffusion_edge\nGrid_a = {}\nGrid_b = {}'.format(grid_a, grid_b))
		self.assertTrue((grid_a == grid_b).all())

	def test_non_local_diffusion_once(self):

		initial_population = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		initial_population[matrix_size//2][matrix_size//2] = 1
		survival_probabilities = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		generator = curandom.XORWOWRandomNumberGenerator()
		run_primitive(Empty_grid().vars(matrix_size) == Initialize_grid().vars(matrix_size, initial_population, survival_probabilities, generator))
		Config.engine.split()
		run_primitive(Non_local_diffusion().vars(non_local_always, matrix_size, p_non_local_always, mu, gamma, grid_dims, block_dims))
		grid_a = Config.engine.stack.pop().get()

		grid_b = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		grid_b[0][0] = 1

		print('Non_local_diffusion_once\nGrid_a = {}\nGrid_b = {}'.format(grid_a, grid_b))
		self.assertTrue((grid_a == grid_b).all())

	def test_non_local_diffusion_twice(self):

		initial_population = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		initial_population[matrix_size//2][matrix_size//2] = 2
		survival_probabilities = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		generator = curandom.XORWOWRandomNumberGenerator()
		run_primitive(Empty_grid().vars(matrix_size) == Initialize_grid().vars(matrix_size, initial_population, survival_probabilities, generator))
		Config.engine.split()
		run_primitive(Non_local_diffusion().vars(non_local_always, matrix_size, p_non_local_always, mu, gamma, grid_dims, block_dims))
		grid_a = Config.engine.stack.pop().get()

		grid_b = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		grid_b[0][0] = 2

		print('Non_local_diffusion_twice\nGrid_a = {}\nGrid_b = {}'.format(grid_a, grid_b))
		self.assertTrue((grid_a == grid_b).all())

	def test_non_local_diffusion_never(self):
		
		initial_population = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		initial_population[matrix_size//2][matrix_size//2] = 1
		survival_probabilities = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		generator = curandom.XORWOWRandomNumberGenerator()
		run_primitive(Empty_grid().vars(matrix_size) == Initialize_grid().vars(matrix_size, initial_population, survival_probabilities, generator))
		Config.engine.split()
		run_primitive(Non_local_diffusion().vars(non_local_never, matrix_size, p_non_local_never, mu, gamma, grid_dims, block_dims))
		grid_a = Config.engine.stack.pop().get()

		grid_b = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		grid_b[matrix_size//2][matrix_size//2] = 1

		print('Non_local_diffusion_never\nGrid_a = {}\nGrid_b = {}'.format(grid_a, grid_b))
		self.assertTrue((grid_a == grid_b).all())

	def test_survival_kernel_none_survive(self):
		
		# make sure all cells die
		initial_population = np.ones((matrix_size,matrix_size)).astype(np.float32)
		survival_probabilities = np.ones((matrix_size,matrix_size)).astype(np.float32)
		generator = curandom.XORWOWRandomNumberGenerator()
		run_primitive(Empty_grid().vars(matrix_size) == Initialize_grid().vars(matrix_size, initial_population, survival_probabilities, generator))
		Config.engine.split()
		run_primitive(Survival_of_the_fittest().vars(survival_none, matrix_size, grid_dims, block_dims))
		grid_a = Config.engine.stack.pop().get()

		grid_b = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		self.assertTrue((grid_a == grid_b).all())

	def test_survival_kernel_all_survive(self):
		
		# make sure no cells die
		initial_population = np.ones((matrix_size,matrix_size)).astype(np.float32)
		survival_probabilities = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		generator = curandom.XORWOWRandomNumberGenerator()
		run_primitive(Empty_grid().vars(matrix_size) == Initialize_grid().vars(matrix_size, initial_population, survival_probabilities, generator))
		Config.engine.split()
		run_primitive(Survival_of_the_fittest().vars(survival_all, matrix_size, grid_dims, block_dims))
		grid_a = Config.engine.stack.pop().get()

		grid_b = np.ones((matrix_size,matrix_size)).astype(np.float32)

		print('Survival_kernel_all_survive\nGrid_a = {}\nGrid_b = {}'.format(grid_a, grid_b))
		self.assertTrue((grid_a == grid_b).all())

	def _test_population_growth_(self):

		initial_population = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		initial_population[matrix_size//2][matrix_size//2] = 1
		initial_population[0][0] = 1
		survival_probabilities = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		generator = curandom.XORWOWRandomNumberGenerator()
		run_primitive(Empty_grid().vars(matrix_size) == Initialize_grid().vars(matrix_size, initial_population, survival_probabilities, generator))
		Config.engine.split()
		run_primitive(Population_growth().vars(population_growth, matrix_size, growth_rate, grid_dims, block_dims))
		grid_a = Config.engine.stack.pop().get()

		grid_b = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		grid_b[matrix_size//2][matrix_size//2] = 227
		grid_b[0][0] = 227
		self.assertTrue((grid_a == grid_b).all())

	def test_get_random_number(self):

		generator = curandom.XORWOWRandomNumberGenerator()
		grid = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		grid = gpuarray.to_gpu(grid)
		for i in range(10):
			get_random_number(generator.state, grid, np.int32(matrix_size), grid = (grid_dims,grid_dims), block = (block_dims,block_dims,1))
			grid_cpu = grid.get()
			for i in range(matrix_size):
				for j in range(matrix_size):
					self.assertGreater(grid_cpu[i][j], 0)
					self.assertLessEqual(grid_cpu[i][j], 1)

	def test_get_random_angle_in_radians(self):
		
		generator = curandom.XORWOWRandomNumberGenerator()
		grid = np.zeros((matrix_size,matrix_size)).astype(np.float32)
		grid = gpuarray.to_gpu(grid)
		for i in range(10):
			get_random_angle(generator.state, grid, np.int32(matrix_size), grid = (grid_dims,grid_dims), block = (block_dims,block_dims,1))
			grid_cpu = grid.get()
			for i in range(matrix_size):
				for j in range(matrix_size):
					self.assertGreater(grid_cpu[i][j], 0)
					self.assertLessEqual(grid_cpu[i][j], 2 * np.pi)

	def test_pop2data2gpu(self):

		grid_a = np.zeros((matrix_size,matrix_size))
		grid_b = np.ones((matrix_size,matrix_size))
		Config.engine.stack.push(grid_a)
		Config.engine.stack.push(grid_b)

		@pop2data2gpu
		def func(a,b):
			np.add(a,b)
			return a,b

		grid_c = np.ones((matrix_size,matrix_size))
		self.assertTrue((Config.engine.stack.pop() == grid_c).all())


# Create the TestBmsb suite        
test_Bmsb_suite = unittest.TestLoader().loadTestsFromTestCase(TestBmsb)
