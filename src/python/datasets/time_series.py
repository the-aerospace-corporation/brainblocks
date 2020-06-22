""" A script that generates data that looks like rocket and satellite telemetry.
The script focuses primarily on transient signals.
"""

__author__ = "Jacob Everist"
__copyright__ = "Copyright 2017, The Aerospace Corporation"
__version__ = "0.0.1"
__maintainer__ = "Jacob Everist"
__email__ = "jacob.s.everist@aero.org"

import numpy as np
import scipy.signal as sig
import scipy.special as spec
from scipy.stats import truncnorm


# HELPER FUNCTIONS

def make_sample_times(secs=1, sample_rate=100):
    return np.linspace(0, secs, sample_rate * secs, endpoint=False)


def semi_circle(sample_times, centerTime, radius, amplitude=1.0, random_state=None):
    """ half-circle with zeros everywhere else """
    h = centerTime

    interm = (1 - np.multiply(sample_times - h, sample_times - h) / (radius * radius)) * (amplitude * amplitude)
    interm[interm < 0] = np.nan
    result = np.sqrt(interm)
    result += bounded_noise(max_noise=0.01, size=result.size, random_state=random_state)
    result = np.nan_to_num(result)
    return result


def add_depression(sample_times, data, radius=0.05, min_error=0.1, bounded_error=0.5, random_state=None):
    centerTime = np.random.choice(sample_times)
    amplitude = ensure_error(min_error=min_error, bounded_error=bounded_error, random_state=random_state)[0]

    depression = semi_circle(sample_times, centerTime, radius, amplitude)

    return data - depression


def create_step_saw_tooth(sample_times, start_time, stop_time):
    height = 1
    mid_time = start_time + (stop_time - start_time) / 2.0
    data_section1 = np.heaviside(sample_times - start_time, 1.0) \
                    * (1 - np.heaviside(sample_times - mid_time, 1.0)) \
                    * (
                            ((2 * height * sample_times) / (stop_time - start_time)) - (
                            (2 * height * start_time) / (stop_time - start_time)))

    data_section2 = np.heaviside(sample_times - mid_time, 1.0) * (1 - np.heaviside(sample_times - stop_time, 1.0)) * (
            ((-height * sample_times) / (stop_time - (((stop_time - start_time) / 2.0) + start_time))) + (
            (height * stop_time) / (stop_time - (((stop_time - start_time) / 2.0) + start_time))))
    return data_section1 + data_section2


def bounded_noise(max_noise=0.1, size=1, random_state=None):
    a, b = -1.0, 1.0
    r = max_noise * truncnorm.rvs(a, b, size=size, random_state=random_state)
    return r


def ensure_error(min_error=0.1, bounded_error=0.1, size=1, random_state=None):
    a, b = 0, 1.0
    r = (bounded_error * truncnorm.rvs(a, b, size=size, random_state=random_state) + min_error) * np.random.choice(
        [1, -1], size=size)

    return r


def triangle(height, start_time, stop_time, num_samples):
    height = np.float(height)
    start_time = np.float(start_time)
    stop_time = np.float(stop_time)

    rising_sample_count = int(np.floor(num_samples / 2))
    falling_sample_count = int(np.ceil(num_samples / 2))

    x1 = np.linspace(start_time, start_time + (stop_time - start_time) / 2.0, num=rising_sample_count)
    y1 = ((2 * height * x1) / (stop_time - start_time)) - ((2 * height * start_time) / (stop_time - start_time))

    x2 = np.linspace(start_time + (stop_time - start_time) / 2.0, stop_time, num=falling_sample_count)
    y2 = ((-height * x2) / (stop_time - (((stop_time - start_time) / 2.0) + start_time))) + (
            (height * stop_time) / (stop_time - (((stop_time - start_time) / 2.0) + start_time)))

    index = np.concatenate((x1, x2), axis=0)
    values = np.concatenate((y1, y2), axis=0)

    return index, values


# GENERATOR FUNCTIONS

def generate_exp_decay(sample_times, **kwargs):
    has_spatial_abnormality = kwargs.get('has_spatial_abnormality', False)
    max_noise = kwargs.get('max_noise', 0.01)

    random_state = kwargs.get('random_state')
    if random_state is not None:
        np.random.seed(random_state)

    result = -np.exp(sample_times)

    result += bounded_noise(max_noise=max_noise, size=result.size, random_state=random_state)

    if has_spatial_abnormality:
        result = add_depression(sample_times, result, random_state=random_state)

    return sample_times, result


def generate_exp_growth(sample_times, **kwargs):
    has_spatial_abnormality = kwargs.get('has_spatial_abnormality', False)
    max_noise = kwargs.get('max_noise', 0.01)

    random_state = kwargs.get('random_state')
    if random_state is not None:
        np.random.seed(random_state)

    result = np.exp(sample_times)
    result += bounded_noise(max_noise=max_noise, size=result.size, random_state=random_state)
    if has_spatial_abnormality:
        result = add_depression(sample_times, result, random_state=random_state)
    return sample_times, result


def generate_signal_to_nom(sample_times, **kwargs):
    has_spatial_abnormality = kwargs.get('has_spatial_abnormality', False)
    max_noise = kwargs.get('max_noise', 0.01)

    random_state = kwargs.get('random_state')
    if random_state is not None:
        np.random.seed(random_state)

    result = np.exp(-sample_times) * np.cos(2 * np.pi * sample_times)
    result += bounded_noise(max_noise=max_noise, size=result.size, random_state=random_state)
    if has_spatial_abnormality:
        result = add_depression(sample_times, result, random_state=random_state)
    return sample_times, result


def generate_ramp(sample_times, **kwargs):
    has_spatial_abnormality = kwargs.get('has_spatial_abnormality', False)
    max_noise = kwargs.get('max_noise', 0.01)

    random_state = kwargs.get('random_state')
    if random_state is not None:
        np.random.seed(random_state)

    startTime = 0.1
    slope = 1.2
    result = np.maximum(slope * (sample_times - startTime), 0.0)
    result += bounded_noise(max_noise=max_noise, size=result.size, random_state=random_state)
    if has_spatial_abnormality:
        result = add_depression(sample_times, result, random_state=random_state)
    return sample_times, result


def generate_sine(sample_times, **kwargs):
    has_spatial_abnormality = kwargs.get('has_spatial_abnormality', False)
    max_noise = kwargs.get('max_noise', 0.01)

    random_state = kwargs.get('random_state')
    if random_state is not None:
        np.random.seed(random_state)

    period = 0.5
    amplitude = 0.5
    result = amplitude * np.sin(2 * np.pi * sample_times / period)
    result += bounded_noise(max_noise=max_noise, size=result.size, random_state=random_state)
    if has_spatial_abnormality:
        result = add_depression(sample_times, result, random_state=random_state)
    return sample_times, result


def generate_sigmoid(sample_times, **kwargs):
    has_spatial_abnormality = kwargs.get('has_spatial_abnormality', False)
    max_noise = kwargs.get('max_noise', 0.01)

    random_state = kwargs.get('random_state')
    if random_state is not None:
        np.random.seed(random_state)

    steepness = 10.0
    offset = 0.5
    result = spec.expit(steepness * (sample_times - offset))
    result += bounded_noise(max_noise=max_noise, size=result.size, random_state=random_state)
    if has_spatial_abnormality:
        result = add_depression(sample_times, result, random_state=random_state)
    return sample_times, result


def generate_square(sample_times, **kwargs):
    has_spatial_abnormality = kwargs.get('has_spatial_abnormality', False)
    max_noise = kwargs.get('max_noise', 0.01)

    random_state = kwargs.get('random_state')
    if random_state is not None:
        np.random.seed(random_state)

    waveFreq = 5
    result = sig.square(2 * np.pi * waveFreq * sample_times)

    result += bounded_noise(max_noise=max_noise, size=result.size, random_state=random_state)

    if has_spatial_abnormality:
        result = add_depression(sample_times, result, random_state=random_state)

    return sample_times, result


def generate_overshoot_sine(sample_times, **kwargs):
    has_spatial_abnormality = kwargs.get('has_spatial_abnormality', False)
    has_time_abnormality = kwargs.get('has_time_abnormality', False)
    has_freq_abnormality = kwargs.get('has_freq_abnormality', False)
    max_noise = kwargs.get('max_noise', 0.01)

    random_state = kwargs.get('random_state')
    if random_state is not None:
        np.random.seed(random_state)

    steepness = 20.0

    randScale = 0.01
    offset = 0.5
    sineFreq = 50 + np.fabs(np.random.normal(scale=2))

    if has_freq_abnormality:
        sineFreq = np.fabs(50 + np.random.normal(scale=2 * 100))

    sineAmp = 0.1 + np.fabs(np.random.normal(scale=randScale))

    wobbleComp = sineAmp * np.sin(sineFreq * (sample_times - offset - 0.12)) * spec.expit(
        100.0 * (sample_times - offset - 0.12)) * (
                         1 - spec.expit(5.0 * (sample_times - offset - 0.16)))
    result = spec.expit(steepness * (sample_times - offset)) * (1 + wobbleComp)

    result += bounded_noise(max_noise=max_noise, size=result.size, random_state=random_state)

    if has_spatial_abnormality:
        result = add_depression(sample_times, result, random_state=random_state)

    return sample_times, result


def generate_sigmoid_then_decay(sample_times, **kwargs):
    has_spatial_abnormality = kwargs.get('has_spatial_abnormality', False)
    has_time_abnormality = kwargs.get('has_time_abnormality', False)
    max_noise = kwargs.get('max_noise', 0.01)

    random_state = kwargs.get('random_state')
    if random_state is not None:
        np.random.seed(random_state)

    startTime = 0.2  # + np.fabs(np.random.normal(scale=randScale))
    decayTime = 0.6 + np.fabs(bounded_noise(max_noise=max_noise, random_state=random_state)[0])

    if has_time_abnormality:
        decayTime = 0.6 + ensure_error(min_error=0.04, bounded_error=0.06, random_state=random_state)[0]

    steepnessOne = 20.0
    steepnessTwo = 0.01

    stepPart = spec.expit(steepnessOne * (sample_times - startTime)) * (1 - np.heaviside(sample_times - decayTime, 0.5))
    decayPart = np.heaviside(sample_times - decayTime, 0.5) * (steepnessTwo / (sample_times - decayTime + steepnessTwo))

    result = stepPart + decayPart
    result += bounded_noise(max_noise=max_noise, size=result.size, random_state=random_state)

    if has_spatial_abnormality:
        result = add_depression(sample_times, result, random_state=random_state)

    return sample_times, result


def generate_step_then_decay(sample_times, **kwargs):
    has_spatial_abnormality = kwargs.get('has_spatial_abnormality', False)
    has_time_abnormality = kwargs.get('has_time_abnormality', False)
    max_noise = kwargs.get('max_noise', 0.01)

    random_state = kwargs.get('random_state')
    if random_state is not None:
        np.random.seed(random_state)

    startTime = 0.05  # + np.fabs(np.random.normal(scale=randScale))
    decayTime = 0.2 + np.fabs(bounded_noise(max_noise=max_noise, random_state=random_state)[0])

    if has_time_abnormality:
        decayTime = 0.2 + ensure_error(min_error=0.04, bounded_error=0.06, random_state=random_state)[0]

    steepness = 0.01

    stepPart = np.heaviside(sample_times - startTime, 0.5) - np.heaviside(sample_times - decayTime, 0.5)
    decayPart = np.heaviside(sample_times - decayTime, 0.5) * (steepness / (sample_times - decayTime + steepness))

    result = stepPart + decayPart
    result += bounded_noise(max_noise=max_noise, size=result.size, random_state=random_state)

    if has_spatial_abnormality:
        result = add_depression(sample_times, result, random_state=random_state)

    return sample_times, result


def generate_multi_square(sample_times, **kwargs):
    has_spatial_abnormality = kwargs.get('has_spatial_abnormality', False)
    has_time_abnormality = kwargs.get('has_time_abnormality', False)
    max_noise = kwargs.get('max_noise', 0.01)

    random_state = kwargs.get('random_state')
    if random_state is not None:
        np.random.seed(random_state)

    delayOne = 0.05  # + np.fabs(np.random.normal(scale=randScale))
    delayTwo = 0.1 + np.fabs(bounded_noise(max_noise=max_noise, random_state=random_state)[0])
    delayThree = 0.1 + np.fabs(bounded_noise(max_noise=max_noise, random_state=random_state)[0])

    lengthOne = 0.2 + np.fabs(bounded_noise(max_noise=max_noise, random_state=random_state)[0])
    lengthTwo = 0.2 + np.fabs(bounded_noise(max_noise=max_noise, random_state=random_state)[0])

    if has_time_abnormality:
        lengthTwo = 0.2 + ensure_error(min_error=0.04, bounded_error=0.06, random_state=random_state)[0]

    lengthThree = 0.2 + np.fabs(bounded_noise(max_noise=max_noise, random_state=random_state)[0])

    startOne = delayOne
    endOne = startOne + lengthOne

    startTwo = delayTwo + endOne
    endTwo = startTwo + lengthTwo

    startThree = delayThree + endTwo
    endThree = startThree + lengthThree

    blockOne = np.heaviside(sample_times - startOne, 0.5) - np.heaviside(sample_times - endOne, 0.5)
    blockTwo = np.heaviside(sample_times - startTwo, 0.5) - np.heaviside(sample_times - endTwo, 0.5)
    blockThree = np.heaviside(sample_times - startThree, 0.5) - np.heaviside(sample_times - endThree, 0.5)

    result = blockOne + blockTwo + blockThree
    result += bounded_noise(max_noise=max_noise, size=result.size, random_state=random_state)

    if has_spatial_abnormality:
        result = add_depression(sample_times, result, random_state=random_state)

    return sample_times, result


def generate_sawtooth_waveform(sample_times, **kwargs):
    has_spatial_abnormality = kwargs.get('has_spatial_abnormality', False)
    has_time_abnormality = kwargs.get('has_time_abnormality', False)
    max_noise = kwargs.get('max_noise', 0.01)

    random_state = kwargs.get('random_state')
    if random_state is not None:
        np.random.seed(random_state)

    delay_time = 0.0
    length_time = 0.2

    start_time = 0.0
    stop_time = start_time + length_time
    # startTime = delayTime + np.fabs(boundedNoise(boundVal=0.01)[0])
    # stopTime = startTime + lengthTime + np.fabs(bounded_noise(max_noise=max_noise)[0])

    result = create_step_saw_tooth(sample_times, start_time, stop_time)

    while stop_time < sample_times[-1]:
        start_time = delay_time + stop_time
        if has_time_abnormality:
            start_time += np.fabs(ensure_error(min_error=0.04, bounded_error=0.06, random_state=random_state)[0])

        stop_time = start_time + length_time
        result += create_step_saw_tooth(sample_times, start_time, stop_time)

    result += bounded_noise(max_noise=max_noise, size=result.size, random_state=random_state)

    if has_spatial_abnormality:
        result = add_depression(sample_times, result, random_state=random_state)

    return sample_times, result


def generate_brownian(sample_times, **kwargs):

    has_spatial_abnormality = kwargs.get('has_spatial_abnormality', False)
    max_noise = kwargs.get('max_noise', 0.01)
    scale = kwargs.get('scale', 1.0)

    # the scale of time step for which the brownian computation operates at
    # the smaller the time step, the higher the resolution the variation occurs
    change_rate = kwargs.get('change_rate', 0.01)

    random_state = kwargs.get('random_state')
    if random_state is not None:
        np.random.seed(random_state)

    # generate brownian motion seed
    # seed = rgen.randint(1000, 1000000)
    seed = np.random.randint(1000, 1000000)

    decay = np.sqrt(0.5)
    depth = 7
    mean = 0.0
    std = 1.0
    correction = 16.0 / 3.0

    def value_at_time(t):
        bias = 0.0

        def seeded_gaussian(offset):
            rstate = np.random.RandomState(seed + offset)
            return rstate.normal(0, 1)

        # FAQ: Where on earth does this 16/3 come from?
        #
        # It comes from the biasAdd term below.  Specifically, if you
        # add up the squares of the coefficients rrOffset-t, t-llOffset,
        # rOffset-t, and t-lOffset (across the entire interval from t to t+1,
        # because the values are interpolated), you get 16/3.  It is twice
        # the integral of t^2 dt from t = 0 to 2.
        #
        # Long story short, it makes the standard deviation of the bias
        # term go to 1 (approximately) over long periods of time, which is
        # what we want.  This is then multiplied by the std parameter to
        # obtain the final values.

        factor = np.sqrt((1.0 - decay ** 2) / correction)

        # We successively generate smaller and smaller biases which together
        # produce the proper standard deviation.  Each integer time value
        # acts as a "tentpole" which contributes a certain amount to the
        # eventual output value, which varies from a strength of 2 exactly
        # at the tentpole, linearly down to 0 at distance 2 from the tentpole.
        # The same process is repeated with the time dimension zoomed in by
        # a factor of <scale>, and the output dimension scaled down by a
        # factor of <decay>.  This happens for <depth> loops.

        for i in range(depth):
            lOffset = int(np.floor(t))
            rOffset = lOffset + 1
            llOffset = lOffset - 1
            rrOffset = rOffset + 1

            l = seeded_gaussian(lOffset)
            r = seeded_gaussian(rOffset)
            ll = seeded_gaussian(llOffset)
            rr = seeded_gaussian(rrOffset)

            biasAdd = (rrOffset - t) * l + (t - llOffset) * r + (rOffset - t) * ll + (t - lOffset) * rr
            bias = bias + biasAdd * factor

            t = scale * t
            factor = factor * decay

        return mean + bias * std

    # scale the time range
    integer_times = sample_times / change_rate

    # apply the computation elementwise to the sample times
    apply_func = np.vectorize(value_at_time)
    result = apply_func(integer_times)

    # fit values from 0 to 1
    result = np.interp(result, (result.min(), result.max()), (0, 1))

    result += bounded_noise(max_noise=max_noise, size=result.size, random_state=random_state)

    if has_spatial_abnormality:
        result = add_depression(sample_times, result, min_error=0.3, bounded_error=0.6,
                                random_state=random_state)

    return sample_times, result


### FIXME:  Unintegrated generators
def states(numstates, numtrans, slope, interval, stopTime):
    stopTime = np.float(stopTime)
    randplat = np.random.choice(range(0, numstates + numtrans), numtrans, replace=False)

    height = 0
    sample_times = np.array([])
    result = np.array([])
    for n in range(0, numstates + numtrans):
        if n in randplat:
            sample_times = np.concatenate((sample_times, [n * stopTime, n * stopTime + interval]))
            result = np.concatenate((result, [height, height]))
        else:
            sample_times = np.concatenate((sample_times, [n * stopTime, n * stopTime + interval]))
            result = np.concatenate((result, [height, height + slope * interval]))
            height += slope * interval

    return sample_times, result


def missing_peak(num_peaks=5, num_anom=0, height=1, sample_rate=500, secs=1, **kwargs):
    """
    startTime = 0.2  # + np.fabs(np.random.normal(scale=randScale))
    decayTime = 0.6 + np.fabs(bounded_noise(max_noise=max_noise)[0])

    if has_time_abnormality:
        decayTime = 0.6 + ensure_error(min_error=0.04, bounded_error=0.06)[0]

    """

    has_spatial_abnormality = kwargs.get('has_spatial_abnormality', False)
    has_time_abnormality = kwargs.get('has_time_abnormality', False)
    has_freq_abnormality = kwargs.get('has_freq_abnormality', False)
    max_noise = kwargs.get('max_noise', 0.01)

    random_state = kwargs.get('random_state')
    if random_state is not None:
        np.random.seed(random_state)

    missing_peaks = np.random.choice(range(0, num_peaks), num_anom, replace=False)

    interval = secs / num_peaks

    samples_per_peak = int(sample_rate * interval)

    sample_count = samples_per_peak * num_peaks

    result = np.array([])

    for n in range(0, num_peaks):
        if n in missing_peaks:
            index, values = triangle(0, n * secs / num_peaks, n * secs / num_peaks + interval,
                                     samples_per_peak)
        else:
            index, values = triangle(height, n * secs / num_peaks, n * secs / num_peaks + interval,
                                     samples_per_peak)
        result = np.concatenate((result, values), axis=0)

    # sample_times = np.linspace(0, secs, num=sample_count)
    sample_times = make_sample_times(secs=secs, sample_rate=sample_rate)

    result += bounded_noise(max_noise=max_noise, size=result.size, random_state=random_state)

    if has_spatial_abnormality:
        result = add_depression(sample_times, result, random_state=random_state)

    return sample_times, result


class FixedSequence(object):

    def __init__(self):
        self.symbols = ["a", "b", "c", "d"]
        self.index = 0
        self.val = "a"

    def __iter__(self):
        self.val = self.symbols[self.index]
        return self

    def __next__(self):
        self.index += 1
        if self.index >= len(self.symbols):
            self.index = 0
        self.val = self.symbols[self.index]

        return self.val


class SignalGenerator(object):

    def __init__(self):
        self.count = 0
        self.val = 0
        self.periodCount = 1

    def __iter__(self):
        self.val = 0
        return self

    def __next__(self):
        self.count += 1

        if self.periodCount % 100 == 0:
            if self.count >= 10:

                self.val += 1
                self.val = self.val % 2
                if self.val == 0:
                    self.periodCount += 1

                self.count = 0
        else:
            if self.count >= 3:

                self.val += 1
                self.val = self.val % 2
                if self.val == 0:
                    self.periodCount += 1

                self.count = 0

        if True:
            return np.float(self.val)
        else:
            raise StopIteration


def Clifford(x, y, a, b, c, d, *o):
    return np.sin(a * y) + c * np.cos(a * x), \
           np.sin(b * x) + d * np.cos(b * y)
    # return x + (np.sin(a * y) + c * np.cos(a * x)) * 0.1, \
    #       y + (np.sin(b * x) + d * np.cos(b * y)) * 0.1

    #    xs[i + 1] = xs[i] + (x_dot * dt)
    #    ys[i + 1] = ys[i] + (y_dot * dt)


def trajectory_coords(fn, x0, y0, a, b=0., c=0., d=0., e=0., f=0., n=1000):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in np.arange(n - 1):
        x[i + 1], y[i + 1] = fn(x[i], y[i], a, b, c, d, e, f)
    return x, y


def lemiscate(n=100, alpha=1.0):
    # t = np.linspace(0, 2*np.pi, num=n)
    sample_times = np.linspace(np.pi / 2, 2 * np.pi + np.pi / 2, num=n)

    x = alpha * np.sqrt(2) * np.cos(sample_times) / (np.sin(sample_times) ** 2 + 1)
    y = alpha * np.sqrt(2) * np.cos(sample_times) * np.sin(sample_times) / (np.sin(sample_times) ** 2 + 1)

    return x, y


def circle(n=100, r=1.0):
    angles = np.linspace(0, 2 * np.pi, num=n)
    xs = r * np.cos(angles)
    ys = r * np.sin(angles)

    # x = []
    # y = []
    # for theta in np.linspace(0, 10 * np.pi):
    #    r = ((theta) ** 2)
    #    x.append(r * np.cos(theta))
    #    y.append(r * np.sin(theta))

    return xs, ys


def spiral(n=100):
    angles = np.linspace(0, 10 * np.pi, num=n)
    xs = 0.002 * (angles ** 2) * np.cos(angles)
    ys = 0.002 * (angles ** 2) * np.sin(angles)

    # x = []
    # y = []
    # for theta in np.linspace(0, 10 * np.pi):
    #    r = ((theta) ** 2)
    #    x.append(r * np.cos(theta))
    #    y.append(r * np.sin(theta))

    return xs, ys


"""
def random_walk():
    random.seed(6)
    n = 10000  # Number of points
    f = filter_width = 5000  # momentum or smoothing parameter, for a moving average filter

    # filtered random walk
    xs = np.convolve(np.random.normal(0, 0.1, size=n), np.ones(f) / f).cumsum()
    ys = np.convolve(np.random.normal(0, 0.1, size=n), np.ones(f) / f).cumsum()

    # Add "mechanical" wobble on the x axis
    # xs += 0.1 * np.sin(0.1 * np.array(range(n - 1 + f)))

    # Add "measurement" noise
    # xs += np.random.normal(0, 0.005, size=n - 1 + f)
    # ys += np.random.normal(0, 0.005, size=n - 1 + f)

    # Add a completely incorrect value
    # xs[int(len(xs) / 2)] = 100
    # ys[int(len(xs) / 2)] = 0

    # Create a dataframe
    df = pd.DataFrame(dict(x=xs, y=ys))

    return df


def trajectory(fn, x0, y0, a, b=0., c=0., d=0., e=0., f=0., n=1000):
    x, y = trajectory_coords(fn, x0, y0, a, b, c, d, e, f, n)
    return pd.DataFrame(dict(x=x, y=y))
"""


def lorenz(x, y, s=10, r=28, b=2.667):
    '''
    Given:
       x, y: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot: values of the lorenz attractor's partial
           derivatives at the point x, y
    '''

    return
    # x_dot = s * (y - x)
    # y_dot = r * x - y - x * z
    # z_dot = x * y - b * z
    # return x_dot, y_dot, z_dot
