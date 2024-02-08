import pathlib
import json

import dataset


def encode(ds, strategy, strategy_type, model_path, to_exclude):
    if strategy == 'AVES':
        ds.encode_aves(model_path, strategy=strategy_type, labels_to_exclude=to_exclude)
    elif strategy == 'CAE':
        ds.encode_ae(model_path, nfft=2048, sample_dur=3, n_mel=128, bottleneck=256,
                     labels_to_exclude=to_exclude, input_type=strategy_type)
    else:
        raise Exception('%s is not defined as a strategy' % strategy)


if __name__ == '__main__':
    # Get the dataset config
    config_path = pathlib.Path(input('Where is the config path of the dataset?'))

    encoding_strategy = input('Which encoding strategy should we use? AVES/CAE ')
    if encoding_strategy == 'CAE':
        input_type = input('Which CAE strategy should we use? fixed/cropsduration/crops ')
    else:
        input_type = input('Which AVES strategy should we use? mean/max ')
    # Transform the detections in features (adding also freq limits and duration)
    f = open(config_path)
    config = json.load(f)
    ds_test = dataset.LifeWatchDataset(config)
    labels_to_exclude = ['boat_sound', 'boat_noise', 'water_movement', 'boat_operations',
                         'electronic_noise', 'interference', 'voice', 'out_of_water', 'deployment']
    encode(ds_test, encoding_strategy, input_type, labels_to_exclude)
