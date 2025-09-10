# pylint: disable=C0301
'''
This module contains the AudioProcessor class and related functions for processing audio data.
It utilizes various libraries and models to perform tasks such as preprocessing, feature extraction,
and audio separation. The class is initialized with configuration parameters and can process
audio files using the provided models.
'''
import math
import os

import librosa
import numpy as np
import torch
from audio_separator.separator import Separator
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor

from data.audio.wav2vec import Wav2VecModel
from data.audio.util import resample_audio


class AudioProcessor:
    """
    AudioProcessor is a class that handles the processing of audio files.
    It takes care of preprocessing the audio files, extracting features
    using wav2vec models, and separating audio signals if needed.

    :param sample_rate: Sampling rate of the audio file
    :param fps: Frames per second for the extracted features
    :param wav2vec_model_path: Path to the wav2vec model
    :param only_last_features: Whether to only use the last features
    :param audio_separator_model_path: Path to the audio separator model
    :param audio_separator_model_name: Name of the audio separator model
    :param cache_dir: Directory to cache the intermediate results
    :param device: Device to run the processing on
    """
    def __init__(
        self,
        sample_rate,
        fps,
        wav2vec_model_path,
        wav2vec_feature_type,
        audio_separator_model_path:str=None,
        audio_separator_model_name:str=None,
        cache_dir:str='',
        device="cuda:0",
    ) -> None:
        self.sample_rate = sample_rate
        self.fps = fps
        self.device = device

        self.audio_encoder = Wav2VecModel.from_pretrained(wav2vec_model_path, local_files_only=True).to(device=device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        assert wav2vec_feature_type in ["mid", "last", "all"]
        self.wav2vec_feature_type = wav2vec_feature_type

        if audio_separator_model_name is not None:
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError as _:
                print("Fail to create the output cache dir.")
            self.audio_separator = Separator(
                output_dir=cache_dir,
                output_single_stem="vocals",
                model_file_dir=audio_separator_model_path,
            )
            self.audio_separator.load_model(audio_separator_model_name)
            assert self.audio_separator.model_instance is not None, "Fail to load audio separate model."
        else:
            self.audio_separator=None
            print("Use audio directly without vocals seperator.")


        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_path, local_files_only=True)


    def preprocess(self, wav_file: str, 
                   clip_length: int=-1, 
                   padding=False,
                   processed_length=0):
        """
        Preprocess a WAV audio file by separating the vocals from the background and resampling it to a 16 kHz sample rate.
        The separated vocal track is then converted into wav2vec2 for further processing or analysis.

        Args:
            wav_file (str): The path to the WAV file to be processed. This file should be accessible and in WAV format.

        Raises:
            RuntimeError: Raises an exception if the WAV file cannot be processed. This could be due to issues
                        such as file not found, unsupported file format, or errors during the audio processing steps.

        Returns:
            torch.tensor: Returns an audio embedding as a torch.tensor
        """
        if self.audio_separator is not None:
            # 1. separate vocals
            # TODO: process in memory
            outputs = self.audio_separator.separate(wav_file)
            if len(outputs) <= 0:
                raise RuntimeError("Audio separate failed.")

            vocal_audio_file = outputs[0]
            vocal_audio_name, _ = os.path.splitext(vocal_audio_file)
            vocal_audio_file = os.path.join(self.audio_separator.output_dir, vocal_audio_file)
            vocal_audio_file = resample_audio(vocal_audio_file, os.path.join(self.audio_separator.output_dir, f"{vocal_audio_name}-16k.wav"), self.sample_rate)
        else:
            vocal_audio_file=wav_file

        # 2. extract wav2vec features
        speech_array, sampling_rate = librosa.load(vocal_audio_file, sr=self.sample_rate)
        audio_feature = np.squeeze(self.wav2vec_feature_extractor(speech_array, sampling_rate=sampling_rate).input_values)
        seq_len = math.ceil(len(audio_feature) / self.sample_rate * self.fps)
        audio_length = seq_len

        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)

        if padding:
            if clip_length>0 and seq_len % clip_length != 0:
                all_len = seq_len + processed_length
                audio_feature = torch.nn.functional.pad(audio_feature, (0, (clip_length - all_len % clip_length) * (self.sample_rate // self.fps)), 'constant', 0.0)
                seq_len += clip_length - all_len % clip_length
        audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=seq_len, output_hidden_states=True)
        assert len(embeddings) > 0, "Fail to extract audio embedding"

        last_hidden = embeddings.last_hidden_state.squeeze()
        mid_hidden = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        mid_hidden = rearrange(mid_hidden, "b s d -> s b d")

        if self.wav2vec_feature_type == "last":
            audio_emb = last_hidden
        elif self.wav2vec_feature_type == "mid":
            audio_emb = mid_hidden
        elif self.wav2vec_feature_type == "all":
            audio_emb = torch.cat([mid_hidden, last_hidden.unsqueeze(1)], dim=1)
        
        audio_emb = audio_emb.cpu().detach()

        return audio_emb, audio_length

    def get_embedding(self, wav_file: str):
        """preprocess wav audio file convert to embeddings

        Args:
            wav_file (str): The path to the WAV file to be processed. This file should be accessible and in WAV format.

        Returns:
            torch.tensor: Returns an audio embedding as a torch.tensor
        """
        speech_array, sampling_rate = librosa.load(
            wav_file, sr=self.sample_rate)
        assert sampling_rate == 16000, "The audio sample rate must be 16000"
        audio_feature = np.squeeze(self.wav2vec_feature_extractor(
            speech_array, sampling_rate=sampling_rate).input_values)
        seq_len = math.ceil(len(audio_feature) / self.sample_rate * self.fps)

        audio_feature = torch.from_numpy(
            audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(
                audio_feature, seq_len=seq_len, output_hidden_states=True)
        assert len(embeddings) > 0, "Fail to extract audio embedding"   

        last_hidden = embeddings.last_hidden_state.squeeze()
        mid_hidden = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        mid_hidden = rearrange(mid_hidden, "b s d -> s b d")

        if self.wav2vec_feature_type == "last":
            audio_emb = last_hidden
        elif self.wav2vec_feature_type == "mid":
            audio_emb = mid_hidden
        elif self.wav2vec_feature_type == "all":
            audio_emb = torch.cat([mid_hidden, last_hidden.unsqueeze(1)], dim=1)

        audio_emb = audio_emb.cpu().detach()

        return audio_emb
    
    def get_audio_emb_window(self, audio_emb, frame_num, frame0_idx, audio_shift=2):
        zero_audio_embed = torch.zeros((audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype)
        zero_audio_embed_3 = torch.zeros((3, audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype)  # device=audio_emb.device
        iter_ = 1 + (frame_num - 1) // 4
        audio_emb_wind = []
        for lt_i in range(iter_):
            if lt_i == 0:  # latent_i
                # 提取第一帧VAElatent，audio左侧补0，标识出
                st = frame0_idx + lt_i - 2
                ed = frame0_idx + lt_i + 3
                wind_feat = torch.stack([
                    audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                    for i in range(st, ed)
                ], dim=0)  # [5, 13, 768]
                wind_feat = torch.cat((zero_audio_embed_3, wind_feat), dim=0)  # [8, 13, 768]
            else:
                st = frame0_idx + 1 + 4 * (lt_i - 1) - audio_shift
                ed = frame0_idx + 1 + 4 * lt_i + audio_shift
                wind_feat = torch.stack([
                    audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                    for i in range(st, ed)
                ], dim=0)  # [8, 13, 768]
            audio_emb_wind.append(wind_feat)
        audio_emb_wind = torch.stack(audio_emb_wind, dim=0)  # [iter_, 8, 13, 768]

        return audio_emb_wind, ed - audio_shift

    def close(self):
        """
        TODO: to be implemented
        """
        return self

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
