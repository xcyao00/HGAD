import numpy as np
import torch


class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)


class LabelAugmentor():
    def __init__(self, mapping=list(range(10))):
        self.mapping = mapping

    def __call__(self, l):
        return int(self.mapping[l])


class Augmentor():
    def __init__(self, deterministic, noise_amplitde, uniform_dequantize, beta, gamma, tanh, ch_pad=0, ch_pad_sig=0):
        self.deterministic      = deterministic
        self.sigma_noise        = noise_amplitde
        self.uniform_dequantize = uniform_dequantize
        self.beta               = beta
        self.gamma              = gamma
        self.tanh               = tanh
        self.ch_pad             = ch_pad
        self.ch_pad_sig         = ch_pad_sig
        assert ch_pad_sig <= 1., 'Padding sigma must be between 0 and 1.'

    def __call__(self, x):
        if not self.deterministic:
            if self.uniform_dequantize:
                x += torch.rand_like(x) / 256.
            if self.sigma_noise > 0.:
                x += self.sigma_noise * torch.randn_like(x)

        x = self.gamma * (x - self.beta)

        if self.tanh:
            x.clamp_(min=-(1 - 1e-7), max=(1 - 1e-7))
            x = 0.5 * torch.log((1+x) / (1-x))

        if self.ch_pad:
            padding = torch.cat([x] * int(np.ceil(float(self.ch_pad) / x.shape[0])), dim=0)[:self.ch_pad]
            padding *= np.sqrt(1. - self.ch_pad_sig**2)
            padding += self.ch_pad_sig * torch.randn(self.ch_pad, x.shape[1], x.shape[2])
            x = torch.cat([x, padding], dim=0)

        return x

    def de_augment(self, x):
        if self.ch_pad:
            x = x[:, :-self.ch_pad]

        if self.tanh:
            x = torch.tanh(x)

        if isinstance(self.gamma, float):
            return x / self.gamma + self.beta
        else:
            return x / self.gamma.to(x.device) + self.beta.to(x.device)


class FeatureAugmentor():
    def __init__(self, deterministic, noise_amplitde, uniform_dequantize):
        self.deterministic      = deterministic
        self.sigma_noise        = noise_amplitde
        self.uniform_dequantize = uniform_dequantize

    def __call__(self, x):
        if not self.deterministic:
            if self.uniform_dequantize:
                x += torch.rand_like(x) / 256.
            if self.sigma_noise > 0.:
                x += self.sigma_noise * torch.randn_like(x)

        return x
