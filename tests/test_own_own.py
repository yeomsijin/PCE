import unittest
import numpy as np
from tpce.own_own import own_own_costs
from tpce.entropy import entropy_excess,direction
class OwnOwnPairBandwidth(unittest.TestCase):
 def test_pair_formula_swap_and_fixed_override(self):
  x=np.array([1.,3.]);y=np.array([2.])
  cost,sigma,meta=own_own_costs(x,y)
  self.assertAlmostEqual(sigma**2,.875)
  rev,other,_=own_own_costs(y,x)
  self.assertAlmostEqual(sigma,other)
  np.testing.assert_allclose(cost,rev[::-1],atol=1e-15)
  self.assertAlmostEqual(float(direction(*cost)),-float(direction(*rev)))
  fixed,s,_=own_own_costs(x,y,bandwidth_method='fixed',bandwidth=.7)
  np.testing.assert_array_equal(fixed,[entropy_excess(x,y,.7),entropy_excess(y,x,.7)])
 def test_degenerate_not_silently_floored(self):
  for x,y in [([],[]),([1],[1])]:
   with self.assertRaises(ValueError):own_own_costs(x,y)
  with self.assertRaises(ValueError):own_own_costs([1],[2],bandwidth=.1)
  costs,_,_=own_own_costs([],[],bandwidth_method='fixed',bandwidth=.1)
  np.testing.assert_array_equal(costs,[0,0])
if __name__=='__main__':unittest.main()
