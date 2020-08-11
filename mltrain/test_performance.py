import unittest
from mltrain.performance import PerformanceCollection, Performance, HigherIsBetterMetric, LowerIsBetterMetric, setup_metrics


class TestPerformance(unittest.TestCase):
    def test_higher_is_better_metric(self):
        higher_is_better_metric = HigherIsBetterMetric('auc')
        p0 = Performance(higher_is_better_metric)
        p1 = Performance(higher_is_better_metric, 1)
        p2 = Performance(higher_is_better_metric, 2)
        self.assertTrue(p1.cmp(p0))
        self.assertTrue(p2.cmp(p0))
        self.assertTrue(p2.cmp(p1))
        self.assertFalse(p0.cmp(p1))
        self.assertFalse(p0.cmp(p2))
        self.assertFalse(p1.cmp(p2))

    def test_lower_is_better_metric(self):
        lower_is_better_metric = LowerIsBetterMetric('mse')
        p0 = Performance(lower_is_better_metric)
        p1 = Performance(lower_is_better_metric, -1)
        p2 = Performance(lower_is_better_metric, -2)
        self.assertTrue(p1.cmp(p0))
        self.assertTrue(p2.cmp(p0))
        self.assertTrue(p2.cmp(p1))
        self.assertFalse(p0.cmp(p1))
        self.assertFalse(p0.cmp(p2))
        self.assertFalse(p1.cmp(p2))

    def test_performance_collection(self):
        m_mse = LowerIsBetterMetric('mse')
        m_auc = HigherIsBetterMetric('auc')
        best_performance = setup_metrics([m_mse, m_auc])

        data = [(10, 0, True),
                ( 9, 1, True),
                (10, 2, False),
                ( 8, 1, True),
                ( 9, 3, False),
                ( 8, 0, False),
                ( 8, 2, True),
                ( 6, 0, True),
                ( 7, 5, False),
                ( 5, 1, True),
                ( 6, 6, False)]

        for mse, auc, target in data:
            perf = best_performance.update({'mse': mse, 'auc': auc})
            if target and perf.cmp(best_performance):
                best_performance = perf
            elif not target and not perf.cmp(best_performance):
                continue
            else:
                self.fail()



