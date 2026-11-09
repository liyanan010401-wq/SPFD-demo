import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).absolute().parents[1]))
import torch, numpy as np
from src.core import *
class TestSPFD(unittest.TestCase):
    def test_real_fft_and_partition(self):
        seed_all(4);x=torch.rand(24,24);m=spectral_masks(x)
        self.assertTrue(torch.equal(sum(m),torch.ones_like(x)))
        for a in m:self.assertLess(torch.fft.ifft2(torch.fft.fft2(x)*a).imag.abs().max(),1e-6)
    def test_restoration_and_padding(self):
        x=torch.rand(24,24);valid=torch.ones_like(x,dtype=torch.bool);valid[-2:]=False
        y,p,_=mix_restore(x,torch.rand_like(x),torch.zeros_like(x),torch.tensor([[.5,.5,.4,.3]]),valid)
        self.assertTrue(torch.equal(y[p==1],x[p==1]));self.assertTrue(torch.equal(y[~valid],x[~valid]))
        self.assertEqual(float(protection(torch.empty(0,4),24,24).sum()),0)
    def test_hungarian_pair_not_query_index(self):
        t=torch.tensor([[.2,.3,.1,.2],[.8,.6,.1,.2]]);b=torch.cat([t,torch.tensor([[.5,.5,.1,.1]])]);l=torch.tensor([3.,4.,-3.]);perm=[1,2,0]
        self.assertLess(float(consistency(l,b,l[perm],b[perm],t)),1e-6)
    def test_donor_permissions(self):
        self.assertEqual(legal_donors(0,[0,1,2],np.array([1,2,3]),np.array([1,1,2])),[1])
    def test_both_view_query_gradients(self):
        b=torch.tensor([[.5,.5,.2,.2]],requires_grad=True);l=torch.tensor([1.],requires_grad=True);l2=torch.tensor([2.],requires_grad=True)
        consistency(l,b,l2,b*.95,b.detach()).backward();self.assertGreater(abs(l.grad.item()),0);self.assertGreater(abs(l2.grad.item()),0)
    def test_regional_kl_stops_only_teacher(self):
        a=torch.tensor([[.8,.2],[.4,.6]],requires_grad=True);b=torch.tensor([[.6,.4],[.2,.8]],requires_grad=True);y=torch.tensor([0,1]);v=torch.ones(2,dtype=torch.bool)
        posterior_loss(a,b,y,v).backward();ga=a.grad.clone();gb=b.grad.clone();a2=a.detach().clone().requires_grad_();b2=b.detach().clone().requires_grad_();ce=-.5*(a2[range(2),y].log()+b2[range(2),y].log()).mean();ce.backward()
        self.assertTrue(torch.allclose(ga,a2.grad));self.assertFalse(torch.allclose(gb,b2.grad))

    def test_source_only_normalization(self):
        raw=torch.tensor([[[[0.,1.],[2.,3.]]],[[[1.,3.],[2.,4.]]],[[[1.,4.],[2.,9.]]]])
        valid=torch.ones_like(raw,dtype=torch.bool)
        a,sa=preprocess(raw,valid,np.array([0,1]))
        changed=raw.clone();changed[2]=torch.tensor([[[-100.,4.],[20.,9000.]]])
        b,sb=preprocess(changed,valid,np.array([0,1]))
        self.assertEqual(sa,sb)
        self.assertTrue(torch.equal(a[:2],b[:2]))
        self.assertLess(float(a[:2].mean().abs()),1e-6)
    def test_spectral_clip_before_inverse_no_pixel_clip(self):
        seed_all(8);x=torch.randn(24,24);valid=torch.ones_like(x,dtype=torch.bool)
        mask=torch.ones_like(x);boxes=torch.empty(0,4)
        actual,p,_=mix_restore(x,torch.randn_like(x),mask,boxes,valid)
        f=torch.fft.fft2(x)
        expected=torch.fft.ifft2(f.abs().clamp(0,2)*torch.exp(1j*torch.angle(f))).real
        self.assertTrue(torch.allclose(actual,expected,atol=1e-7))
        self.assertLess(float(actual.min()),0.)
        self.assertEqual(float(p.sum()),0.)
    def test_query_box_sum4(self):
        b0=torch.tensor([[.4,.4,.2,.2]])
        b1=b0+.1
        value=consistency(torch.tensor([1.]),b0,torch.tensor([1.]),b1,b0)
        self.assertAlmostEqual(float(value),.4,places=6)
    def test_shared_concentration_initialization_and_no_cap(self):
        head=RegionalHead()
        self.assertAlmostEqual(float(F.softplus(head.raw_k).detach()),.974076984,places=6)
        with torch.no_grad():
            head.proj.weight.zero_();head.proj.bias.zero_();head.proj.bias[0]=1
            head.directions.zero_();head.directions[0,0]=1;head.directions[1,1]=1
            head.raw_k.fill_(25.)
        q,valid=head(torch.zeros(1,16))
        self.assertTrue(bool(valid[0]))
        self.assertAlmostEqual(float((q[0,0]/q[0,1]).log().detach()),25.,places=4)
    def test_single_layer_and_both_view_feature_gradients(self):
        model=TinyDetector();head=RegionalHead()
        _,_,feature=model(torch.randn(2,1,24,24));feature.retain_grad()
        self.assertEqual(feature.shape[1],16)
        self.assertEqual(head.proj.in_features,16)
        rois=torch.tensor([[.2,.2,.5,.5]])
        a,va=head(roi_vectors(feature[:1],rois));b,vb=head(roi_vectors(feature[1:],rois))
        posterior_loss(a,b,torch.tensor([1]),va&vb).backward()
        self.assertGreater(float(feature.grad[0].abs().sum()),0.)
        self.assertGreater(float(feature.grad[1].abs().sum()),0.)
    def test_empty_regional_loss_is_graph_zero(self):
        a=torch.empty(0,2,requires_grad=True);b=torch.empty(0,2,requires_grad=True)
        loss=posterior_loss(a,b,torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.bool))
        self.assertEqual(float(loss.detach()),0.)
        loss.backward()
        self.assertIsNotNone(a.grad);self.assertIsNotNone(b.grad)
    def test_small_norm_region_excluded(self):
        head=RegionalHead()
        with torch.no_grad():head.proj.weight.zero_();head.proj.bias.zero_()
        q,valid=head(torch.zeros(2,16))
        self.assertFalse(bool(valid.any()))
        loss=posterior_loss(q,q,torch.tensor([0,1]),valid)
        self.assertEqual(float(loss.detach()),0.)

if __name__=='__main__':unittest.main()
