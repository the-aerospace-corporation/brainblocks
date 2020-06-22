import os
import matplotlib.pyplot as plt
import pandas as pd

from brainblocks.datasets.time_series import make_sample_times, generate_overshoot_sine, generate_step_then_decay, \
    generate_sigmoid_then_decay, generate_multi_square, generate_sawtooth_waveform, generate_exp_decay, \
    generate_exp_growth, generate_ramp, generate_sigmoid, generate_signal_to_nom, generate_sine, generate_square, \
    generate_brownian

# TODO: discrete states
# TODO: data gaps
# TODO: corruption
# TODO: variable sample rate
# TODO: parameterize all parts of data generation and adulteration: abnormalities, errors, noise

fileRootName = "test_telemetry"
iter_fmt_str = "_%04u"
fileName = fileRootName + iter_fmt_str
outputDir = "test_dir"


# FIXME: only some of the generators implement time abnormalities

# different flavors of the data
flavors = ["training", "nominal", "abnormal_depression", "abnormal_time"]

# the current data generators
dataGenerators = dict(overshoot_sine=generate_overshoot_sine,
                      step_then_decay=generate_step_then_decay,
                      multi_square=generate_multi_square,
                      sigmoid_then_decay=generate_sigmoid_then_decay,
                      exp_decay=generate_exp_decay,
                      exp_growth=generate_exp_growth,
                      ramp=generate_ramp,
                      sigmoid=generate_sigmoid,
                      signal_to_nom=generate_signal_to_nom,
                      sine=generate_sine,
                      square=generate_square,
                      brownian=generate_brownian,
                      sawtooth_waveform=generate_sawtooth_waveform
                      )

# whether to generate a PNG plot
do_plots = True

# how many data sets per flavor
num_sets = 2

# sample times
secs = 1
sample_rate = 100
sample_times = make_sample_times(secs=secs, sample_rate=sample_rate)

# maximum noise added to data
max_noise = 0.01
gen_params = dict(max_noise=max_noise)

# for each generator, generate the data
for generator_name, generator_func in dataGenerators.items():

    # for each flavor, generate the data
    for flavorName in flavors:

        # update filename with signature for string formatting
        dirName = outputDir + "/" + generator_name + "/" + flavorName

        # create directory if it doesn't exist
        if not os.path.exists(dirName):
            os.makedirs(dirName)

        # generate the data num_sets number of times
        for k in range(num_sets):

            if flavorName == "nominal" or flavorName == "training":
                index, values = generator_func(sample_times, random_state=k, **gen_params)
            elif flavorName == "abnormal_time":
                index, values = generator_func(sample_times, random_state=k, has_time_abnormality=True, **gen_params)
            elif flavorName == "abnormal_depression":
                index, values = generator_func(sample_times, random_state=k, has_spatial_abnormality=True, **gen_params)
            else:
                raise Exception("unknown flavor")

            # put data into pandas dataframe
            sigDF = pd.DataFrame(index=index, data=values, columns=["value"])

            # save to CSV file
            sigDF.to_csv(dirName + "/" + fileName % k + ".csv", index=True, index_label="time")

            # generate plot of data
            if do_plots:
                sigDF.plot()
                plt.savefig(dirName + "/" + fileName % k + ".png")
                plt.close()
