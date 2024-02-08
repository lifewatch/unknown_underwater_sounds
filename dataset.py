import datetime
import json
import os
import pathlib
import shutil
import sys

import fairseq
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import scipy
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from PIL import Image
# from maad import util
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import train_test_split
import copy
import suntime
import pytz
import noisereduce as nr

from transformers import ClapModel, ClapProcessor
from transformers import pipeline

import models
import utils as u

torchaudio.set_audio_backend(backend='soundfile')

# matplotlib.use('TkAgg')
# Get the color map by name:
cm = plt.get_cmap('jet')


class LifeWatchDataset:
    def __init__(self, config):
        # Spectrogram settings
        self.desired_fs = config['desired_fs']
        self.channel = config['channel']
        self.log = config['log']
        self.color = config['color']

        # Folders
        self.wavs_folder = pathlib.Path(config['wavs_folder'])

        self.annotations_file = config['annotations_file']

        self.nfft = config['nfft']
        self.win_len = config['win_len']
        self.hop_length = int(self.win_len / config['hop_ratio'])
        self.win_overlap = self.win_len - self.hop_length

        self.normalization_style = config['normalization_style']

        self.MIN_DURATION = config['min_duration']
        self.MAX_DURATION = config['max_duration']
        self.MIN_SNR = 10

        self.config = config

    def __setitem__(self, key, value):
        if key in self.config.keys():
            self.config[key] = value
        self.__dict__[key] = value

    def save_config(self, config_path):
        with open(config_path, 'w') as f:
            json.dump(self.config, f)

    def all_snippets(self, detected_foregrounds, labels_to_exclude=None):

        file_list = os.listdir(self.wavs_folder)
        for i, row in tqdm(detected_foregrounds.iterrows(), total=len(detected_foregrounds)):
            wav_path = row['wav']
            waveform_info = torchaudio.info(wav_path)

            # If the selection is in between two files, open both and concatenate them
            if row['Beg File Samp (samples)'] > row['End File Samp (samples)']:
                waveform1, fs = torchaudio.load(wav_path,
                                                frame_offset=row['Beg File Samp (samples)'],
                                                num_frames=waveform_info.num_frames - row[
                                                    'Beg File Samp (samples)'])

                wav_path2 = self.wavs_folder.joinpath(file_list[file_list.index(row['Begin File']) + 1])
                waveform2, fs = torchaudio.load(wav_path2,
                                                frame_offset=0,
                                                num_frames=row['End File Samp (samples)'])
                waveform = torch.cat([waveform1, waveform2], -1)
            else:
                waveform, fs = torchaudio.load(wav_path,
                                               frame_offset=row['Beg File Samp (samples)'],
                                               num_frames=row['End File Samp (samples)'] - row[
                                                   'Beg File Samp (samples)'])
            if waveform_info.sample_rate > self.desired_fs:
                waveform = F.resample(waveform=waveform, orig_freq=fs, new_freq=self.desired_fs)[self.channel, :]
            else:
                waveform = waveform[self.channel, :]

            yield i, row, waveform

    def load_relevant_selection_table(self, labels_to_exclude=None):
        annotations_file = pathlib.Path(self.annotations_file)
        if annotations_file.is_dir():
            selections_list = list(annotations_file.glob('*.txt'))
        else:
            selections_list = [annotations_file]
        for selection_table_path in selections_list:
            print('Annotations table %s' % selection_table_path.name)
            selections = pd.read_table(selection_table_path)
            if 'Tags' in selections.columns:
                if labels_to_exclude is not None:
                    selections = selections.loc[~selections.Tags.isin(labels_to_exclude)]

            # Filter the selections
            selections = selections.loc[selections['Low Freq (Hz)'] < (self.desired_fs / 2)]
            selections = selections.loc[selections.View == 'Spectrogram 1']
            if 'SNR NIST Quick (dB)' in selections.columns:
                selections = selections.loc[selections['SNR NIST Quick (dB)'] > self.MIN_SNR]
            selections.loc[selections['High Freq (Hz)'] > (self.desired_fs / 2), 'High Freq (Hz)'] = self.desired_fs / 2
            selections = selections.loc[
                (selections['End Time (s)'] - selections['Begin Time (s)']) >= self.MIN_DURATION]
            selections = selections.loc[
                (selections['End Time (s)'] - selections['Begin Time (s)']) <= self.MAX_DURATION]

            yield selection_table_path, selections

    def convert_raven_to_ae_format(self, labels_to_exclude=None):
        total_encoder = pd.DataFrame()
        for _, selection_table in self.load_relevant_selection_table(labels_to_exclude=labels_to_exclude):
            for wav_file, wav_selections in selection_table.groupby('Begin File'):
                wav_path = list(self.wavs_folder.glob('**/' + wav_file))[0]
                wav = sf.SoundFile(wav_path)
                duration = wav_selections['End Time (s)'] - wav_selections['Begin Time (s)']
                two_files = wav_selections['Beg File Samp (samples)'] > wav_selections['End File Samp (samples)']
                pos = wav_selections['Beg File Samp (samples)'] / wav.samplerate + duration / 2
                encoder_df = pd.DataFrame({'begin_sample': wav_selections['Beg File Samp (samples)'],
                                           'end_sample': wav_selections['End File Samp (samples)'],
                                           'pos': pos,
                                           'duration': duration,
                                           'filename': wav_selections['Begin File'],
                                           'label': wav_selections['Tags'],
                                           'min_freq': wav_selections['Low Freq (Hz)'],
                                           'max_freq': wav_selections['High Freq (Hz)'],
                                           'two_files': two_files})
                if 'SNR NIST Quick (dB)' in wav_selections.columns:
                    encoder_df['snr'] = wav_selections['SNR NIST Quick (dB)']
                total_encoder = pd.concat([total_encoder, encoder_df])

        return total_encoder

    def encode_aves(self, model_path, strategy='mean', labels_to_exclude=None):
        output_name = 'AVES_features_space_%s' % strategy
        features_path = self.dataset_folder.joinpath(output_name + '.pt')
        if not features_path.exists():
            model_list, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
            model = model_list[0]
            model.feature_extractor.requires_grad_(False)
            model.eval()
            features_list = []
            idxs = []
            for _, detected_foregrounds in self.load_relevant_selection_table(labels_to_exclude=labels_to_exclude):
                detected_foregrounds['height'] = detected_foregrounds[
                                                     'High Freq (Hz)'] - detected_foregrounds['Low Freq (Hz)']
                detected_foregrounds['width'] = detected_foregrounds[
                                                    'End Time (s)'] - detected_foregrounds['Begin Time (s)']

                for i, _, waveform in self.all_snippets(detected_foregrounds, labels_to_exclude=labels_to_exclude):
                    if strategy == 'mean':
                        features = model.extract_features(waveform.expand(1, -1))[0].mean(dim=1)
                    elif strategy == 'max':
                        features = model.extract_features(waveform.expand(1, -1))[0].max(dim=1).values
                    else:
                        raise Exception('Strategy %s is not defined. Only mean or max' % strategy)

                    features_list.append(features.squeeze(dim=0).detach().numpy())
                    idxs.append(i)

            features_space = torch.Tensor(np.stack(features_list).astype(float))
            torch.save(features_space, features_path)
            df = pd.DataFrame(features_space.numpy())
            df.index = idxs
            columns = ['Low Freq (Hz)', 'High Freq (Hz)', 'height', 'width', 'SNR NIST Quick (dB)', 'Tags']
            if 'SNR NIST Quick (dB)' not in detected_foregrounds.columns:
                columns = ['Low Freq (Hz)', 'High Freq (Hz)', 'height', 'width', 'Tags']
            total_df = pd.merge(df, detected_foregrounds[columns],
                                left_index=True, right_index=True)
            total_df = total_df.rename(
                columns={'Low Freq (Hz)': 'min_freq', 'High Freq (Hz)': 'max_freq', 'height': 'bandwidth',
                         'width': 'duration', 'SNR NIST Quick (dB)': 'snr',
                         'Tags': 'label'})

            total_df.to_pickle(self.dataset_folder.joinpath(output_name + '.pkl'))
        else:
            total_df = pd.read_pickle(self.dataset_folder.joinpath(output_name + '.pkl'))

        return

    def encode_ae(self, model_path, nfft, sample_dur, n_mel, bottleneck, labels_to_exclude=None, input_type='fixed'):
        self.MIN_DURATION = nfft / self.desired_fs
        features_path = self.dataset_folder.joinpath('CAE_%s_%s_%s_%s_%s_features_space.pkl' %
                                                     (input_type, nfft, sample_dur, n_mel, bottleneck))
        if not features_path.exists():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if input_type == 'fixed':
                frontend = models.frontend(sr=self.desired_fs, nfft=nfft, sampleDur=sample_dur, n_mel=n_mel).to(device)
            elif input_type == 'cropsduration':
                frontend = models.frontend_crop_duration(sr=self.desired_fs, nfft=nfft,
                                                         sampleDur=sample_dur, n_mel=n_mel).to(device)
            elif input_type == 'crops':
                frontend = models.frontend_crop()

            encoder = models.sparrow_encoder(bottleneck // (n_mel // 32 * 4), (n_mel // 32, 4))
            decoder = models.sparrow_decoder(bottleneck, (n_mel // 32, 4))
            model = torch.nn.Sequential(encoder, decoder).to(device)

            model.load_state_dict(torch.load(model_path))
            model.eval()

            detections = self.convert_raven_to_ae_format(labels_to_exclude=labels_to_exclude)

            if input_type == 'fixed':
                annotations_ds = u.Dataset(df=detections, audiopath=str(self.wavs_folder), sr=self.desired_fs,
                                           sampleDur=sample_dur, channel=self.channel)
            elif input_type == 'cropsduration':
                annotations_ds = u.DatasetCropsDuration(detections, str(self.wavs_folder), self.desired_fs,
                                                        winsize=nfft, n_mel=n_mel, win_overlap=self.win_overlap,
                                                        sampleDur=sample_dur)
            elif input_type == 'crops':
                annotations_ds = u.DatasetCrops(detections, str(self.wavs_folder), self.desired_fs,
                                                winsize=nfft, n_mel=n_mel, win_overlap=self.win_overlap,
                                                sampleDur=sample_dur)

            loader = torch.utils.data.DataLoader(annotations_ds, batch_size=16, shuffle=False,
                                                 num_workers=0, collate_fn=u.collate_fn)
            encodings, idxs = [], []
            with torch.no_grad():
                for x, name in tqdm(loader, leave=False):
                    label = frontend(x.to(device))
                    encoding = model[:1](label)
                    idxs.extend(name.numpy())
                    encodings.extend(encoding.cpu().detach())
            encodings = np.stack(encodings)
            features_space = torch.Tensor(np.stack([encodings]).astype(float))
            features_space = pd.DataFrame(features_space.numpy()[0])
            idxs = np.stack(idxs)
            features_space.index = idxs
            columns = ['min_freq', 'max_freq', 'duration', 'label', 'snr']
            if 'snr' not in detections.columns:
                columns = ['min_freq', 'max_freq', 'duration', 'label']
            total_features = pd.merge(features_space, detections[columns],
                                      left_index=True, right_index=True)
            total_features['bandwidth'] = total_features['max_freq'] - total_features['min_freq']
            total_features.to_pickle(features_path)

        else:
            total_features = pd.read_pickle(features_path)

        return total_features

    def train_ae(self, sample_dur=5, bottleneck=48, n_mel=128, nfft=2048, input_type='fixed'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = models.sparrow_encoder(bottleneck // (n_mel // 32 * 4), (n_mel // 32, 4))
        decoder = models.sparrow_decoder(bottleneck, (n_mel // 32, 4))
        model = torch.nn.Sequential(encoder, decoder).to(device)

        if input_type == 'fixed':
            frontend = models.frontend(sr=self.desired_fs, nfft=nfft, sampleDur=sample_dur, n_mel=n_mel).to(device)
        elif input_type == 'cropsduration':
            frontend = models.frontend_crop_duration(sr=self.desired_fs, nfft=nfft,
                                                     sampleDur=sample_dur, n_mel=n_mel).to(device)
        elif input_type == 'crops':
            frontend = models.frontend_crop()

        # training / optimisation setup
        lr, wdL2, batch_size = 0.00001, 0.0, 64 if torch.cuda.is_available() else 16
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=wdL2, lr=lr, betas=(0.8, 0.999))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .99 ** epoch)
        vgg16 = models.vgg16.eval().to(device)
        loss_fun = torch.nn.MSELoss()

        detections = self.convert_raven_to_ae_format(labels_to_exclude=None)

        if input_type == 'fixed':
            annotations_ds = u.Dataset(df=detections, audiopath=str(self.wavs_folder), sr=self.desired_fs,
                                       sampleDur=sample_dur, channel=self.channel)
        elif input_type == 'cropsduration':
            annotations_ds = u.DatasetCropsDuration(detections, str(self.wavs_folder), self.desired_fs,
                                                    winsize=nfft, n_mel=n_mel, win_overlap=self.win_overlap,
                                                    sampleDur=sample_dur)
        elif input_type == 'crops':
            annotations_ds = u.DatasetCrops(detections, str(self.wavs_folder), self.desired_fs,
                                            winsize=nfft, n_mel=n_mel, win_overlap=self.win_overlap,
                                            sampleDur=sample_dur)

        loader = torch.utils.data.DataLoader(annotations_ds, batch_size=16,
                                             shuffle=False, num_workers=0, collate_fn=u.collate_fn)

        modelname = f'CAE_{input_type}_{bottleneck}_mel{n_mel}'
        step, writer = 0, SummaryWriter(str(self.dataset_folder.joinpath(modelname)))
        print(f'Go for model {modelname} with {len(detections)} vocalizations')
        for epoch in range(100):
            for x, name in tqdm(loader, desc=str(epoch), leave=False):
                optimizer.zero_grad()
                label = frontend(x.to(device))
                x = encoder(label)
                pred = decoder(x)
                vgg_pred = vgg16(pred.expand(pred.shape[0], 3, *pred.shape[2:]))
                vgg_label = vgg16(label.expand(label.shape[0], 3, *label.shape[2:]))

                score = loss_fun(vgg_pred, vgg_label)
                score.backward()
                optimizer.step()
                writer.add_scalar('loss', score.item(), step)

                if step % 50 == 0:
                    images = [(e - e.min()) / (e.max() - e.min()) for e in label[:8]]
                    grid = make_grid(images)
                    writer.add_image('target', grid, step)
                    writer.add_embedding(x.detach(), global_step=step, label_img=label)
                    images = [(e - e.min()) / (e.max() - e.min()) for e in pred[:8]]
                    grid = make_grid(images)
                    writer.add_image('reconstruct', grid, step)

                step += 1
                if step % 500 == 0:
                    scheduler.step()
            torch.save(model.state_dict(), str(self.dataset_folder.joinpath(modelname + '.weights')))
        return str(self.dataset_folder.joinpath(modelname + '.weights'))
