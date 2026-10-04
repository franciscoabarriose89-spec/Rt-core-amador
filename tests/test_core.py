import unittest

from rt_core_amador import (
    AmadorTensor,
    AmadorTransform,
    KahlerManifold,
    NavierStokesOptimizer,
)


class AmadorTensorTests(unittest.TestCase):
    def test_elementwise_arithmetic_and_shape(self):
        tensor = AmadorTensor([[1, 2], [3, 4]])
        self.assertEqual((tensor * 2 + 1).to_list(), [[3.0, 5.0], [7.0, 9.0]])
        self.assertEqual(tensor.shape, (2, 2))

    def test_dot_and_matrix_multiplication(self):
        self.assertEqual(AmadorTensor([1, 2]).dot(AmadorTensor([3, 4])), 11.0)
        result = AmadorTensor([[1, 2], [3, 4]]).matmul(AmadorTensor([[2], [1]]))
        self.assertEqual(result.to_list(), [[4.0], [10.0]])

    def test_rejects_ragged_input_and_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            AmadorTensor([[1], [2, 3]])
        with self.assertRaises(ValueError):
            AmadorTensor([1, 2]) + AmadorTensor([1])


class AmadorTransformTests(unittest.TestCase):
    def test_round_trip_with_phase(self):
        values = [1.0, -2.0, 0.5, 4.0]
        transform = AmadorTransform(phase=0.37)
        restored = transform.inverse(transform.forward(values))
        for actual, expected in zip(restored, values):
            self.assertAlmostEqual(actual.real, expected)
            self.assertAlmostEqual(actual.imag, 0.0)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            AmadorTransform().forward([])


class NavierStokesTests(unittest.TestCase):
    def test_constant_periodic_field_remains_constant(self):
        optimizer = NavierStokesOptimizer(viscosity=0.1, pressure_iterations=10)
        u = [[1.0, 1.0], [1.0, 1.0]]
        v = [[0.0, 0.0], [0.0, 0.0]]
        next_u, next_v, _ = optimizer.step(u, v, 0.01)
        self.assertEqual(next_u, u)
        self.assertEqual(next_v, v)

    def test_rejects_inconsistent_grid_dimensions(self):
        with self.assertRaises(ValueError):
            NavierStokesOptimizer().step([[0.0]], [[0.0, 1.0]], 0.1)


class KahlerManifoldTests(unittest.TestCase):
    def test_default_potential_has_identity_metric(self):
        manifold = KahlerManifold(2)
        metric = manifold.metric_at((1 + 2j, -0.5j))
        self.assertAlmostEqual(metric[0][0].real, 1.0, places=5)
        self.assertAlmostEqual(metric[1][1].real, 1.0, places=5)
        self.assertAlmostEqual(abs(metric[0][1]), 0.0, places=5)
        self.assertAlmostEqual(manifold.norm_squared((0j, 0j), (3 + 0j, 4j)), 25.0, places=5)

    def test_metric_includes_mixed_coordinate_derivatives(self):
        manifold = KahlerManifold(
            2,
            potential=lambda point: abs(point[0] + point[1]) ** 2,
        )
        metric = manifold.metric_at((0j, 0j))
        for row in metric:
            for entry in row:
                self.assertAlmostEqual(entry.real, 1.0, places=5)
                self.assertAlmostEqual(entry.imag, 0.0, places=5)

    def test_rejects_wrong_point_dimension(self):
        with self.assertRaises(ValueError):
            KahlerManifold(2).metric_at((0j,))


if __name__ == "__main__":
    unittest.main()
