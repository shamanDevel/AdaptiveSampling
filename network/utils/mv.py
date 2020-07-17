class MeanVariance():
    """Online algorithm to compute the mean and variance"""

    def __init__(self):
        self.n_ = 0
        self.mean_ = 0
        self.sn_ = 0
        self.lastmean_ = 0

    def append(self, x):
        """
        Appends the new value 'x' into the running mean and variance computation.
        """
        self.n_ += 1
        self.lastmean_ = self.mean_
        self.mean_ += (x-self.lastmean_)/self.n_
        if self.n_ == 1:
            self.sn_ = 0
        else:
            self.sn_ += (x-self.lastmean_)*(x-self.mean_)

    def mean(self):
        """Returns the current mean"""
        return self.mean_
    def var(self):
        """Returns the current variance"""
        return self.sn_ / self.n_
    def count(self):
        """Returns the current number of points"""
        return self.n_

if __name__== "__main__":
    import numpy as np
    import unittest

    class Test(unittest.TestCase):
        def test_against_numpy(self):
            lengths = [1, 2, 5, 20, 1000]
            for length in lengths:
                xs = np.random.random(length)
                # expected
                mean_np = np.mean(xs)
                var_np = np.var(xs)
                # actual
                mv = MeanVariance()
                for x in xs:
                    mv.append(x)
                self.assertEqual(mv.count(), length)
                self.assertAlmostEqual(mv.mean(), mean_np)
                self.assertAlmostEqual(mv.var(), var_np)

    unittest.main()