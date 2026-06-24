"""
Tests for rmsnorm_backward() in utils/pc_utils.py

rmsnorm_backward() manually computes dE/dx (the gradient of the energy w.r.t.
the pre-norm input x) by applying the chain rule through RMSNorm. These tests
verify that the derivation is mathematically correct using three independent
strategies:

  1. Autograd check    — PyTorch's own .backward() gives the reference gradient.
                         Our result must match it to 7 decimal places.

  2. Numerical check   — Computes the gradient from scratch using only arithmetic
                         (no autograd, no PyTorch internals). Our result must agree.

  3. Integration check — When rmsnorm_backward is wired into the real step_linear
                         (fc1 path), one update step must reduce prediction energy.
                         This confirms the gradient points in the correct direction
                         inside the actual PC learning loop.

Run:  python tests/test_rmsnorm_backward.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from utils.pc_utils import rmsnorm_backward, step_linear


# ── shared helper ─────────────────────────────────────────────────────────────

def _make_inputs(B=2, S=4, H=8, seed=42):
    """
    Return a reproducible (x, norm, g) triplet used by all tests.

      x    — pre-norm input tensor, shape [B, S, H]
      norm — nn.RMSNorm with non-trivial random gamma (not all-ones),
             so the gamma scaling path in the backward is exercised
      g    — simulated incoming gradient dE/dy, shape [B, S, H]

    float64 is used throughout so numerical precision is not the bottleneck.
    """
    torch.manual_seed(seed)
    x    = torch.randn(B, S, H, dtype=torch.float64)
    norm = nn.RMSNorm(H, dtype=torch.float64)
    nn.init.uniform_(norm.weight, 0.5, 1.5)   # non-trivial gamma
    g    = torch.randn(B, S, H, dtype=torch.float64)
    return x, norm, g


# ── Test 1: autograd agreement ────────────────────────────────────────────────

class TestAutograd(unittest.TestCase):
    """
    Primary correctness test.

    PyTorch's autograd engine is used as the reference: it computes dE/dx
    through nn.RMSNorm via .backward() and stores the result in x.grad.
    Our rmsnorm_backward() must produce the same tensor.

    The scalar energy proxy used is  E = sum(g * norm(x)).
    This is the standard trick for testing vector-valued gradient functions:
    the gradient of this E w.r.t. x is exactly  g · d(norm(x))/dx,
    which is what rmsnorm_backward(x, norm, g) should return.

    Pass condition: max element-wise error < 1e-7 (7 decimal places).
    """

    def test_matches_pytorch_autograd(self):
        """Our manual gradient must agree with PyTorch autograd to 1e-7."""
        x, norm, g = _make_inputs()

        # --- reference: PyTorch autograd path ---
        # x.clone() makes an independent copy of x so the original is not modified.
        # .detach() cuts any existing gradient history so autograd starts fresh here.
        # .requires_grad_(True) tells PyTorch to track all operations on x_auto
        # so that when we call .backward() it knows how to compute dE/dx.
        x_auto = x.clone().detach().requires_grad_(True)

        # Define the scalar energy E = sum(g * norm(x_auto)).
        # Splitting into two lines makes the intent clearer:
        #   first build E, then ask PyTorch to differentiate it.
        E = (g * norm(x_auto)).sum()
        E.backward()   # PyTorch walks the computation graph and fills x_auto.grad

        expected = x_auto.grad.clone()   # dE/dx computed by autograd, shape [2, 4, 8]

        # --- our manual backward ---
        # x.clone().detach() gives a plain copy with no gradient tracking —
        # rmsnorm_backward uses only raw tensor arithmetic, not autograd.
        actual = rmsnorm_backward(x.clone().detach(), norm, g)   # shape [2, 4, 8]

        # largest element-wise discrepancy between the two gradients
        max_err = (actual - expected).abs().max().item()
        self.assertAlmostEqual(
            max_err, 0.0, places=7,
            msg=f"Max error vs autograd: {max_err:.2e}  (should be < 1e-7)"
        )


# ── Test 2: numerical finite-difference check ─────────────────────────────────

class TestNumerical(unittest.TestCase):
    """
    A second, fully independent correctness check that does not use autograd at all.

    The idea is: if you slightly increase one element of x and measure how
    much the output changes, you get an approximation of the gradient for that
    element. Repeating this for every element gives the full gradient numerically.
    This is called the finite-difference (or numerical gradient) method and comes
    from the basic calculus definition of a derivative:

        df/dx[i]  =  lim_{ε→0}  (f(x[i]+ε) - f(x[i])) / ε

    In practice we use the symmetric (central) version for better accuracy:

        df/dx[i]  ≈  ( f(x[i] + ε) − f(x[i] − ε) ) / 2ε

    where  f(x) = sum(g * norm(x))  is the scalar energy proxy and  ε = 1e-5.

    Why is this test valuable?
    The autograd test (Test 1) proves our result matches PyTorch — but what if
    PyTorch itself were wrong, or what if we misunderstood what autograd computes?
    The finite-difference test is a ground-truth check that uses nothing but
    addition and division. If our rmsnorm_backward agrees with both autograd AND
    finite differences, the derivation is confirmed from two completely independent
    angles.

    A small tensor [1, 2, 4] is used because we loop over every element (slow).
    Pass condition: relative error < 1e-4  (finite differences have inherent
    approximation error from the finite ε, so exact equality is not expected).
    """

    def test_matches_finite_differences(self):
        """Our gradient must agree with the numerical finite-difference gradient."""
        x, norm, g = _make_inputs(B=1, S=2, H=4, seed=7)
        eps = 1e-5

        # Build the numerical gradient by perturbing each element of x one at a time.
        # num_grad[i] will hold the approximate df/dx[i] for each element.
        num_grad = torch.zeros_like(x)
        for i in range(x.numel()):
            xp, xm = x.clone(), x.clone()
            xp.view(-1)[i] += eps          # x with element i nudged up by ε
            xm.view(-1)[i] -= eps          # x with element i nudged down by ε
            # central-difference approximation for this single element
            num_grad.view(-1)[i] = ((g * norm(xp)).sum() - (g * norm(xm)).sum()) / (2 * eps)

        # our analytic gradient from the derivation
        actual = rmsnorm_backward(x.clone().detach(), norm, g)

        # relative error: how large is the discrepancy as a fraction of the gradient?
        # e.g. if num_grad[i] = 0.5 and actual[i] = 0.50003, rel_err ≈ 0.00006
        rel_err = (actual - num_grad).abs().max() / (num_grad.abs().max() + 1e-12)
        self.assertLess(
            rel_err.item(), 1e-4,
            msg=f"Finite-difference relative error: {rel_err:.2e}  (threshold 1e-4)"
        )


# ── Test 3: integration — energy must decrease after x update ─────────────────

class TestIntegration(unittest.TestCase):
    """
    End-to-end sanity check inside the real PC learning loop.

    In step_linear(fc1), the forward path applies  x_input = layer_norm(x)
    before computing  mu = GELU(fc1(x_input)).  The x update uses
    rmsnorm_backward to translate the gradient from x_norm-space back to
    x-space before nudging x.

    This test verifies that after one update step, the squared prediction
    error  ||target - mu||²  is smaller than before — i.e. x moved in the
    direction that reduces energy, not away from it.

    Weight updates are disabled (requires_update=False) to isolate the
    effect of rmsnorm_backward on the x update alone.

    Pass condition: energy_after < energy_before.
    """

    def test_fc1_x_update_reduces_energy(self):
        """One fc1 step with rmsnorm_backward must reduce squared prediction error."""
        torch.manual_seed(0)
        B, S, H_in, H_out = 1, 4, 8, 32

        layer      = nn.Linear(H_in, H_out, dtype=torch.float64)
        layer_norm = nn.RMSNorm(H_in, dtype=torch.float64)
        target     = torch.randn(B, S, H_out, dtype=torch.float64)
        x          = torch.randn(B, S, H_in,  dtype=torch.float64)

        # prediction error before the x update
        energy_before = ((target - F.gelu(layer(layer_norm(x)))) ** 2).sum().item()

        # run one PC inference step — only x is updated, weights are frozen
        x_new, _, _ = step_linear(
            t=0, T=3, target=target, x=x.clone(),
            layer=layer, lateral_conn=None, layer_type="fc1",
            local_lr=0.01, inference_lr=0.1, clamp_value=3.0,
            energy_fn_name="pc_e", requires_update=False,
            td_err=None, layer_norm=layer_norm, optimizer=None,
        )

        # prediction error after the x update
        energy_after = ((target - F.gelu(layer(layer_norm(x_new)))) ** 2).sum().item()

        self.assertLess(
            energy_after, energy_before,
            msg=(
                f"x update did NOT reduce energy — rmsnorm_backward pushed x the wrong way.\n"
                f"  Before: {energy_before:.6f}   After: {energy_after:.6f}"
            )
        )


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
