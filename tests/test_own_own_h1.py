import unittest
import numpy as np
from tpce import own_own_costs,h1_pair_bandwidth,H1_METHOD
from tpce.entropy import entropy_excess,induced_probability,direction


class H1PairBandwidth(unittest.TestCase):
    def test_equal_pd_mass_and_lifetime_weighted_coordinate_moments(self):
        x=np.array([[1.,1.],[3.,3.]]);y=np.array([[2.,2.]])
        # Q masses are 1/8,3/8,1/2, so variance=7/16 for both axes.
        sigma,meta=h1_pair_bandwidth(x,y)
        np.testing.assert_allclose(sigma**2,[.875,.875],rtol=1e-14)
        self.assertEqual(meta['nonempty_diagrams'],2)
        cost,sigma,meta=own_own_costs(x,y)
        reverse,other,_=own_own_costs(y,x)
        np.testing.assert_allclose(sigma,other,atol=1e-15)
        np.testing.assert_allclose(cost,reverse[::-1],atol=1e-15)
        self.assertEqual(meta['estimator_method'],H1_METHOD)

    def test_constant_birth_reduces_exactly_to_existing_h0_kernel(self):
        a=np.array([.1,.3,.6]);b=np.array([.2,.4])
        c0,s0,_=own_own_costs(a,b)
        x=np.column_stack([np.full(len(a),.7),a]);y=np.column_stack([np.full(len(b),.7),b])
        c1,s1,meta=own_own_costs(x,y)
        self.assertEqual(s1[0],0.)
        self.assertEqual(meta['constant_axes'],[0])
        self.assertAlmostEqual(s1[1],s0,places=15)
        np.testing.assert_allclose(c1,c0,rtol=1e-14,atol=1e-16)
        p,logq,extra=induced_probability(x,y,s1,zero_bandwidth_policy='constant_axis')
        self.assertAlmostEqual(np.exp(logq).sum()+extra,1.,places=14)
        self.assertAlmostEqual(float(-p@logq+p@np.log(p)),c1[0],places=14)

    def test_empty_singleton_constant_axes_and_identical_diagrams(self):
        empty=np.empty((0,2));x=np.array([[.3,.5]])
        cost,sig,_=own_own_costs(empty,x)
        np.testing.assert_array_equal(sig,[0.,0.])
        expected=(.5/1.5)**2/50
        np.testing.assert_allclose(cost,[expected,expected],rtol=1e-14)
        self.assertAlmostEqual(float(direction(*cost)),0.,places=14)
        cost,sig,_=own_own_costs(empty,empty)
        np.testing.assert_array_equal(cost,[0.,0.]);self.assertTrue(np.isnan(direction(*cost)))
        cost,_,_=own_own_costs(x,x);np.testing.assert_array_equal(cost,[0.,0.])
        # Zero-lifetime bars never add probability mass or affect the bandwidth.
        cost2,sig2,_=own_own_costs(np.array([[100.,0.]]),x)
        np.testing.assert_array_equal(sig2,[0.,0.]);np.testing.assert_allclose(cost2,[expected,expected])

    def test_constant_lifetime_and_explicit_kernel_reference(self):
        x=np.array([[.1,.5]]);y=np.array([[.9,.5]])
        cost,sig,_=own_own_costs(x,y)
        self.assertEqual(sig[1],0.)
        kernel=np.exp(-.5*(.8/sig[0])**2)
        expected=((.5/1.5)*(kernel-1))**2/50
        np.testing.assert_allclose(cost,[expected,expected],rtol=1e-14)

    def test_invalid_zero_width_is_not_silently_accepted(self):
        x=np.array([[.1,.2]]);y=np.array([[.2,.3]])
        with self.assertRaises(ValueError):entropy_excess(x,y,[0.,0.])
        with self.assertRaises(ValueError):entropy_excess(x,y,[0.,.1],zero_bandwidth_policy='constant_axis')
        with self.assertRaises(ValueError):own_own_costs([1.],y)
        with self.assertRaises(ValueError):own_own_costs(x,y,bandwidth_method='fixed',bandwidth=.1)
        with self.assertRaises(ValueError):own_own_costs(x,y,bandwidth=.1)
        with self.assertRaises(ValueError):own_own_costs([1.],[1.])

if __name__=='__main__':unittest.main()
