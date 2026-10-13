import json
import unittest
import numpy as np
from tpce import TPCEConfig,Calibration,calibrate,estimate
from tpce.model import persistence_variance_bandwidth
from tpce.geometry import embed,neighbor_indices
from tpce.persistence import diagrams

class BandwidthCalibration(unittest.TestCase):
    def test_equal_diagram_mass_and_exact_pair_moment(self):
        ds=[np.array([1.,3.]),np.array([2.])]
        # Q weights: [1/8,3/8,1/2], not uniform weighting of the three bars.
        values=np.array([1.,3.,2.]);weights=np.array([.125,.375,.5])
        expected=np.sum(weights[:,None]*weights[None,:]*(values[:,None]-values[None,:])**2)
        sigma,meta=persistence_variance_bandwidth(ds)
        self.assertAlmostEqual(sigma*sigma,expected,places=14)
        self.assertAlmostEqual(meta['mixture_mean'],2.25)
        replicated,_=persistence_variance_bandwidth([np.tile(ds[0],7),ds[1]])
        self.assertAlmostEqual(sigma,replicated)
        scaled,_=persistence_variance_bandwidth([3*d for d in ds])
        self.assertAlmostEqual(scaled,3*sigma)
        # Variation between PDs must remain even when each has one bar.
        between,_=persistence_variance_bandwidth([[1.],[3.]])
        self.assertAlmostEqual(between,np.sqrt(2))

    def test_empty_degenerate_and_large_offset(self):
        sigma,meta=persistence_variance_bandwidth([[0,1],[3],[],[0,0]])
        self.assertAlmostEqual(sigma,np.sqrt(2));self.assertEqual(meta['empty_diagrams'],2)
        for ds in [[[],[0]],[[2],[2,2]]]:
            with self.assertRaises(ValueError):persistence_variance_bandwidth(ds)
        sigma,_=persistence_variance_bandwidth([[1e12],[1e12+2]])
        self.assertAlmostEqual(sigma,np.sqrt(2))
        with self.assertRaises(ValueError):persistence_variance_bandwidth([[-1,2]])

    def test_default_uses_current_k_all_four_clouds_and_freezes(self):
        data=np.random.default_rng(410).normal(size=(150,2))
        cfg=TPCEConfig(embedding_dimension=2,delay=1,neighbors=7,theiler_full_window=2,calibration_anchors=6,scale_sample_size=40,normalization='none')
        sp=embed(data,2,1);cal=calibrate(sp,cfg)
        self.assertEqual(cfg.bandwidth_method,'persistence_variance')
        self.assertEqual(cal.metadata['calibration_k'],7)
        self.assertEqual(cal.metadata['nonempty_diagrams'],24)
        values=[];weights=[]
        aa=np.array(cal.metadata['anchors'])
        for target in [0,1]:
            nn=neighbor_indices(sp[target],aa,7,cfg.window)
            for anchor,indices in zip(aa,nn):
                for space in [target,1-target]:
                    h0,_=diagrams(sp[space][np.r_[anchor,indices]],cal.scales[space]);h0=h0[h0>0]
                    values.extend(h0);weights.extend(h0/h0.sum()/24)
        values=np.array(values);weights=np.array(weights)
        exact=np.sum(weights[:,None]*weights[None,:]*(values[:,None]-values[None,:])**2)
        self.assertAlmostEqual(cal.h0**2,exact,places=12)
        result=estimate(data,cfg,anchors=np.array([20,70]),calibration=cal)
        self.assertIs(result.calibration,cal)
        fitted=estimate(data,cfg,anchors=np.array([20,70]))
        np.testing.assert_allclose(result.cost,fitted.cost,equal_nan=True)
        other=calibrate(sp[::-1],cfg)
        self.assertAlmostEqual(other.h0,cal.h0,places=14)
        swapped=estimate(data[:,::-1],cfg,anchors=np.array([20,70]),calibration=other)
        np.testing.assert_allclose(swapped.S_PCE,-result.S_PCE,atol=1e-12,equal_nan=True)
        encoded=json.loads(json.dumps(dict(scales=cal.scales,h0=cal.h0,birth=cal.birth,lifetime=cal.lifetime,metadata=cal.metadata)))
        self.assertEqual(Calibration.from_dict(encoded),cal)

    def test_legacy_opt_in_and_validation(self):
        cfg=TPCEConfig(embedding_dimension=2,delay=1,neighbors=7,calibration_k=9,calibration_anchors=6,scale_sample_size=40,normalization='none',bandwidth_method='legacy_local_std')
        sp=embed(np.random.default_rng(411).normal(size=(150,2)),2,1)
        cal=calibrate(sp,cfg)
        self.assertEqual(cal.metadata['method'],'legacy_local_std');self.assertEqual(cal.metadata['calibration_k'],9)
        self.assertGreater(cal.h0,0)
        for args in [dict(calibration_k=0),dict(calibration_k=True),dict(bandwidth_method='median')]:
            with self.assertRaises(ValueError):TPCEConfig(**args)
        self.assertEqual(Calibration.from_dict(dict(scales=[1,1],h0=.1)).metadata,{})

if __name__=='__main__':unittest.main()
