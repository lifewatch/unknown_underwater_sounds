import pathlib
import json

import dataset


def train(ds, input_type):
    ds.train_ae(sample_dur=3, bottleneck=256, n_mel=128, nfft=2048, input_type=input_type)


if __name__ == '__main__':
    # Get the dataset config
    config_path = pathlib.Path(input('Where is the config path of the dataset?'))

    input_type = input('Which CAE strategy should we use? fixed/cropsduration/crops ')

    # Train CAE
    f = open(config_path)
    config = json.load(f)
    ds_test = dataset.LifeWatchDataset(config)
    train(ds_test, input_type)
