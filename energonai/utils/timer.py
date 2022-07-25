import os
import sys
import time

import torch

_GLOBAL_TIMERS = None


class _Timer:
    """Timer."""

    def __init__(self, name, ignore_first):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()
        if ignore_first:
            self.times = 2
        else:
            self.times = 0

    def start(self):
        """Start the timer."""
        if(self.times != 0):
            self.times = self.times - 1
        else: 
            assert not self.started_, 'timer has already been started'
            torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

    def stop(self):
        """Stop the timer."""
        if(self.times != 0):
            self.times = self.times - 1
        else: 
            assert self.started_, 'timer is not started'
            torch.cuda.synchronize()
            self.elapsed_ += (time.time() - self.start_time)
            self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self, ignore_first):
        self.timers = {}
        self.ignore_first = ignore_first

    def __call__(self, name):        
        if name not in self.timers:
            self.timers[name] = _Timer(name, self.ignore_first)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + '-time', value, iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string0 = ''
        string1 = ''
        for name in names:
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            string0 += ' : {}'.format(name)
            string1 += ' : {:.2f}'.format(elapsed_time)

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1):
                print(f'{string0} \n {string1}', flush=True)
        else:
            print(f'{string0} \n {string1}', flush=True)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)


def _set_timers(ignore_first):
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers(ignore_first)


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def get_timers(ignore_first = False):
    """Return timers."""
    if _GLOBAL_TIMERS is None:
        _set_timers(ignore_first)
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS
