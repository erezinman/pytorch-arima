from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from torch_arima.numerical_utils import causal_transposed_conv1d, polymul, polypow


class ARIMAX_RNN(torch.nn.Module):
    def __init__(self, orders: Tuple[int, int, int], n_exog: int = 0, drift: bool = True, dtype=torch.float64):
        super().__init__()

        p, d, q = self._orders = orders
        self._drift = drift
        self._n_exog = n_exog

        self._ar = torch.nn.Parameter(data=torch.zeros(1, 1, p + 1, dtype=dtype, requires_grad=True))
        self._ma = torch.nn.Parameter(data=torch.zeros(1, 1, q, dtype=dtype, requires_grad=True))
        self._x = torch.nn.Parameter(data=torch.zeros(1, 1, n_exog, dtype=dtype, requires_grad=True))
        self._drift = torch.nn.Parameter(data=torch.zeros(1, 1, 1, dtype=dtype, requires_grad=drift))

        # Set memories:
        self._ar_state = torch.nn.Parameter()  # This is the remembered outputs (for AR)
        self._ma_memory = torch.nn.Parameter()  # This is the remembered inputs (for MA)

        self._init_polynomial_coefficients()

        # Defaults:
        self.set_batch_size(1)

        # Init weights
        for t in (self._ar.data, self._ma.data):
            torch.nn.init.xavier_normal_(t, torch.nn.init.calculate_gain('linear'))
        if drift:
            torch.nn.init.normal_(self._drift.data, 0, 1)

        # Fix first parameters to 1
        self._set_last_constant(self._ar, 1)

    @property
    def ar_state(self) -> torch.Tensor:
        return self._ar_state.detach()[:, :, :-1]

    @property
    def ma_memory(self) -> torch.Tensor:
        return self._ma_memory.detach()

    def set_batch_size(self, n: int):
        self._batch_size = n

        p, d, q = self._orders
        dtype = self._drift.data.dtype

        # Reset memory
        self.set_memory(
            ar_state=torch.zeros(n, 1, p + d, dtype=dtype, requires_grad=False),
            ma_memory=torch.zeros(n, 1, q, dtype=dtype, requires_grad=False)
        )

    def set_memory(self, ar_state: torch.Tensor, ma_memory: torch.Tensor):
        # TODO: Check sizes

        # Make the _ar_state's data padded with additional 0 in its end so the contribution of the weights of the last
        # value will always yield zero. This is because the polynomial multiplication $ AR(L|\phi_i) I(L|d) X_t $
        # (see below) will always result in the expression of the form $ 1 \times X_t + f(X_{t-1}, ..., X_{t-...}) $
        # so if we "replace $ X_t $ with 0" (which is what we do here), we just get an evaluation of the
        # $ f(X_{t-1}, ..., X_{t-...}) $ part which is what we want to calculate.

        self._ar_state.data = F.pad(ar_state, [0, 1]).requires_grad_(False)
        self._ma_memory.data = torch.clone(ma_memory).requires_grad_(False)

    def _init_polynomial_coefficients(self):
        p, d, q = self._orders
        self._derivation_coefficients = polypow(torch.as_tensor([-1, 1]), d)

    @staticmethod
    def _set_last_constant(param, const):
        with torch.no_grad():
            param.data[..., -1] = const
        param.register_hook(ARIMAX_RNN._zero_last_grad)

    @staticmethod
    def _zero_last_grad(grad: torch.Tensor):
        grad[..., -1] = 0

    def forward(self, y: torch.Tensor, exog: Optional[torch.Tensor] = None, *,
                update_memory: bool = True) -> torch.Tensor:

        # TODO: check y & exog sizes

        if self.training:
            return self._forward_bulk(y, exog, update_memory=update_memory)
        else:
            return self.forward_once(y, exog, update_memory=update_memory)

    def forward_once(self, y: torch.Tensor, exog: Optional[torch.Tensor] = None, *,
                     update_memory: bool = True) -> torch.Tensor:
        """
        INPUT:  `y` dimensions:    (B, 1, 1)
                `exog` dimensions: (B, X, 1)
        OUTPUT: dimensions:        (B, 1, 1)

        Note that y's index should be 1 before the exogenous-variables' index.
        i.e, x_t = x_t(x_t-1, exog_t).

        """

        ma_memory = self._ma_memory
        ar_state = self._ar_state

        if not update_memory:
            ma_memory = torch.clone(ma_memory.data)
            ar_state = torch.clone(ar_state.data)

        ma_memory[:, :, :-1] = ma_memory[:, :, 1:]
        ma_memory[:, :, -1:] = y[:, :, -1:]

        exog = y[:, :, -1:]
        result = self._drift + torch.conv1d(exog, self._x) + \
                 torch.conv1d(ma_memory, self._ma) + torch.conv1d(ar_state, self._ar[:, :, :-1])

        # Don't update the 0 at the end. See the comment in `set_memory` for more details.
        ar_state[:, :, :-2] = ar_state[:, :, 1:-1]
        ar_state[:, :, -2:-1] = result
        assert result

    def _forward_bulk(self, y: torch.Tensor, exog: Optional[torch.Tensor] = None, *,
                      update_memory: bool = True) -> torch.Tensor:
        """
        INPUT:  `y` dimensions:    (B, 1, T)
                `exog` dimensions: (B, X, T)
        OUTPUT: dimensions:        (B, 1, ...)  TODO

        Where B = Batch Size, X = # exogenous variables, T = Time Series Length.
        """

        p, d, q = self._orders
        assert y.ndim == 3, 'A single-dimensional tensor is expected'
        assert len(y) > p + d + q, f'Can not differentiate {self._d} times a sequence of length {len(y)}.'
        assert torch.is_floating_point(y), f'Expected a floating-point tensor. Got {y.dtype!r}.'

        # The computed function is this:
        # $$ AR(L|\phi_i) I(L|d) X_t = drift + MA(L|\theta_j) \epsilon_t $$
        # Where
        #   X_t is the output/estimation at time t
        #   \epsilon_t is the t error term.
        # and AR, I, and MA are the following functions of the lag operator L (i.e. shift past forward once):
        #   AR(L|\phi_i) = (1 - \sum_{i=1)^{p} \phi_i L^i)
        #   I(L|d) = (1 - L)^d
        #   MA(L|\theta_j) = (1 - \sum_{j=1)^{q} \theta_j L^j)

        # Calculate the part that doesn't depend on the past observations
        ma = causal_transposed_conv1d(y, self._ma)
        result = self._drift + ma + causal_transposed_conv1d(exog, self._x)
        if update_memory:
            self._ma_memory[:, :, -ma.shape[2]:] = ma[:, :, -q:]

        if p + d == 0:
            return result

        ar_state = self._ar_state
        if not update_memory:
            ar_state = torch.clone(ar_state)

        # TODO: SARIMAX's convolutions should be considered to be implemented sparsely!!! (for large S's)

        # Calculate $ AR(L|\phi_i) \times I(L|d) $
        lag_convolution_parameters = polymul(self._ar, self._derivation_coefficients)
        assert len(lag_convolution_parameters) == p + d + 1, 'Failed sanity check!'

        for t in range(y):
            result[:, :, t] += torch.conv1d(ar_state, lag_convolution_parameters)[:, :, -1]
            ar_state[:, :, :-2] = ar_state[:, :, 1:-1]
            ar_state[:, :, -2] = result[:, :, t]

        return result
