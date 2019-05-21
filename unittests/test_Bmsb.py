
from forest import *
import unittest
import numpy as np

class TestBmsb(unittest.Testcase):

	def setUp(self):
		Config.engine = cuda_engine

	def test_cuda_engine(self):

		self.assertFalse(Config.engine.continue_cycle)
		self.assertEqual(Config.engine.n_iters, 0)
		self.assertEqual(Config.engine.iters, 0)

	def test_stack(self):
		
		mylist = [1,2]
		stack = Stack()
		stack.push(1)
		stack.push(2)
		self.assertListEqual(stack, mylist)
		self.assertTrue(stack.notempty())
		self.assertEqual(stack.pop(), 2)
		self.assertEqual(stack.pop(), 1)
		self.assertFalse(stack.notempty())

	def test_queue(self):
		
		mylist = [1,2]
		queue = Queue()
		queue.enqueue(2)
		queue.enqueue(1)
		self.assertListEqual(queue, mylist)
		self.assertTrue(queue.notempty())
		self.assertEqual(queue.dequeue(), 2)
		self.assertEqual(queue.dequeue(), 1)
		self.assertFalse(queue.notempty())

	def test_grid_initialization(self):

		grid_a = initialize_grid.size(10)
		grid_b = np.zeros((10,10)).astype(np.float32)
		grid_b[5][5] = 1
		self.assertListEqual(grid_a, grid_b)

		grid_c = empty_grid.size(10)
		grid_d = np.zeros((10,10)).astype(np.float32)
		self.assertListEqual(grid_c, grid_d)

	def test_bsmb_stop_condition(self):

		bsmb_stop_condition.vars(3)
		self.assertEqual(Config.engine.n_iters, 3)

	def test_bmsb_stop(self):

		Config.engine.continue_cycle = True
		for i in range(5):
			bmsb_stop
		self.assertEqual(Config.engine.iters, 5)
		self.assertFalse(Config.engine.continue_cycle)

	def test_local_diffusion_kernel(self):

		#assume P_LOCAL = 1.0 (diffusion always occurs)
		grid_a = initialize_grid.size(10)
		grid_b = empty_grid.size(10)
		local_diffusion(play_bmsb.LOCAL, 5, 2)

		grid_c = np.zeros((10,10)).astype(np.float32)
		for i in range(4,7):
			for j in range(4,7):
				grid_c[i][j] = 1
		self.assertListEqual(grid_a, grid_c)

		#assume P_LOCAL = 0.0 (diffusion never occurs)
		grid_a = initialize_grid.size(10)
		grid_b = empty_grid.size(10)
		non_local_diffusion(play_bmsb.NON_LOCAL, 5, 2)

		grid_c = np.zeros((10,10)).astype(np.float32)
		grid_c[5][5] = 1
		self.assertListEqual(grid_a, grid_c)

		#edge case
		grid_a = empty_grid.size(10)
		grid_a[0][0] = 1
		grid_b = empty_grid.size(10)
		local_diffusion(play_bmsb.LOCAL, 5, 2)

		grid_c = np.zeros((10,10))
		grid_c[0][0] = 1
		self.assertListEqual(grid_a, grid_c)

	def test_non_local_diffusion_kernel(self):
		pass

	def test_survival_kernel(self):
		pass

	def test_random_number_generation(self):
		pass

	def test_split(self):
		pass

	def test_merge(self):
		pass

	def test_cycle_start(self):
		pass

	def test_cycle_termination(self):
		pass

	def test_pop2data2gpu(self):

		grid_a = np.zeros((10,10))
		grid_b = np.ones((10,10))
		Config.engine.stack.push(arr1)
		Config.engine.stack.push(arr2)

		@pop2data2gpu
		def func(a,b):
			np.add(a,b)
			return a,b

		grid_c = np.ones((10,10))
		self.assertListEqual(Config.engine.stack.pop(), grid_c)


# Create the TestBmsb suite        
test_Bmsb_suite = unittest.TestLoader().loadTestsFromTestCase(TestBmsb)
