# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import tarfile
from argparse import ArgumentParser
from pathlib import Path
from time import time

import librosa
import numpy as np
import requests
import sox
from ctcdecode_numpy import SeqCtcLmDecoder
from openvino.tools.pot import (
    DataLoader,
    IEEngine,
    Metric,
    compress_model_weights,
    create_pipeline,
    load_model,
)
from openvino.tools.pot.utils.logger import get_logger, init_logger


# Initialize the logger to print the quantization process in the console.
init_logger(level="INFO")
logger = get_logger(__name__)


class LibriSpeechDataLoader(DataLoader):
    def __init__(self, config):
        super().__init__(config)
        self._collect_data()

        self._sampling_rate = 16_000
        self._window_size = 512
        self._mel_fmin = 20.0
        self._mel_fmax = 8000.0

        self._sox_tfm = sox.Transformer()
        self._sox_tfm.set_output_format(
            file_type="wav",
            rate=self._sampling_rate,
            bits=16,
            channels=1,
            encoding="signed-integer",
        )

        self._mel_basis = librosa.filters.mel(
            sr=16_000,
            n_fft=512,
            n_mels=40,
            fmin=self._mel_fmin,
            fmax=self._mel_fmax,
            norm=None,
            htk=True,
        )

        self._context_len = 18
        self._block_len = 16
        self._left_padding = 9
        self._right_padding = 9

    def _collect_data(self):
        self._audio_paths = list(Path(self.config.data_source).glob("**/*.flac"))

        annotation_paths = list(Path(self.config.data_source).glob("**/*.txt"))
        self._annotations = dict()
        for annotation_path in annotation_paths:
            with open(annotation_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                id, label = line.strip().split(" ", 1)
                self._annotations[id] = label

    def __getitem__(self, index):
        audio_path = self._audio_paths[index]

        # convert flac to wav
        audio = self._sox_tfm.build_array(str(audio_path))
        # normalize -1 to 1
        audio = audio / np.float32(32768)

        spec = np.abs(
            librosa.core.spectrum.stft(
                audio,
                n_fft=512,
                hop_length=320,
                win_length=512,
                center=False,
                window="hann",
                pad_mode="reflect",
            )
        )
        # match tf: zero spectrum below fmin/sr*2*(n_fft-1)
        freq_bin_fmin = round(
            self._mel_fmin / self._sampling_rate * 2 * (self._window_size - 1)
        )
        spec[: freq_bin_fmin + 1, :] = 0.0

        melspectrum = np.dot(self._mel_basis, spec)

        # match tf: use np.log() instead of power_to_db() to get correct normalization
        mfcc = librosa.feature.mfcc(
            S=np.log(melspectrum + 1e-30),
            norm="ortho",
            n_mfcc=26,
        )
        # match tf: un-correct 0-th bin normalization
        mfcc[0] *= 2**0.5
        mfcc = mfcc.T

        mfcc = np.vstack((np.zeros((self._left_padding, 26)), mfcc))
        align_right_len = (
            -(len(mfcc) + self._right_padding - self._context_len)
        ) % self._block_len
        pad_right_len = self._right_padding + align_right_len
        if pad_right_len > 0:
            mfcc = np.vstack((mfcc, np.zeros((pad_right_len, 26))))

        # Create data at each time step
        # Note that only first elements of h_states, c_states are given as initial hidden states
        # and subsequent elements are None which are fetched from model output at previous tiem steps
        # based on 'recurrent_out_in_map' passed to the IEEngine.
        input_nodes = []
        h_states = []
        c_states = []
        for idx, start_pos in enumerate(
            range(
                self._context_len,
                len(mfcc) - self._block_len + 1,
                self._block_len,
            )
        ):

            feature = mfcc[start_pos - self._context_len : start_pos + self._block_len]
            input_node = np.lib.stride_tricks.as_strided(
                feature,
                (
                    self._block_len,
                    self._context_len + 1,
                    26,
                ),
                (
                    feature.strides[0],
                    feature.strides[0],
                    feature.strides[1],
                ),
                writeable=False,
            )
            input_nodes.append(input_node)
            if idx == 0:
                h_states.append(np.zeros((1, 2048)))
                c_states.append(np.zeros((1, 2048)))
            else:
                h_states.append(None)
                c_states.append(None)

        item_data = dict(
            input_node=input_nodes,
            previous_state_c=c_states,
            previous_state_h=h_states,
        )

        audio_id = audio_path.stem
        annotation = self._annotations[audio_id]
        item_annotation = (index, annotation)

        return (
            item_annotation,
            item_data,
            {"audio_path": audio_path},
        )

    def __len__(self):
        return len(self._audio_paths)


class WER(Metric):
    def __init__(self, decoder):
        super().__init__()
        self._name = "WER"
        self._decoder = decoder
        self._sum_wer = 0
        self._sum_words = 0
        self._cur_wer = 0

        self._tokens = list(" abcdefghijklmnopqrstuvwxyz'")

        self._cache_probs = dict()
        self._cache_labels = dict()

    @property
    def value(self):
        """Returns metric value for the last model output."""
        return dict(
            WER=self._cur_wer,
        )

    @property
    def avg_value(self):
        """Returns metric value for all model outputs."""
        return dict(
            WER=self._sum_wer / self._sum_words if self._sum_words != 0 else 0,
        )

    def update(self, outputs, targets, metas):
        """
        Note that if 'recurrent_out_in_map' is used, POT returns additional metadata
        , which are "_batch_id", "_cur_time_step", "_max_time_step",
        indicating where outputs are belong to.
        For example, if metadata is {
            "_batch_id": 2,
            "_cur_time_step": 7,
            "_max_times_step": 20,
        } then 'outputs' are model output of 2nd batch at 7th time step input

        :param output: model output
        :param target: annotations for model output
        :param metas: metadata for batch data
        """

        batch_id = metas[0]["_batch_id"]
        if batch_id not in self._cache_labels:
            self._cache_labels[batch_id] = targets

        if batch_id not in self._cache_probs:
            self._cache_probs[batch_id] = []

        probs = outputs[0].squeeze()
        self._cache_probs[batch_id].append(probs)

        cur_time_step = metas[0]["_cur_time_step"]
        max_time_step = metas[0]["_max_time_step"]

        if cur_time_step == max_time_step - 1:
            targets = self._cache_labels.pop(batch_id)
            probs = self._cache_probs.pop(batch_id)

            for prob in probs:
                self._decoder.append_data(prob, log_probs=False)

            hypotheses = [i.text for i in self._decoder.decode(finalize=True)]
            targets = [target.lower() for target in targets]

            for hypo, target in zip(hypotheses, targets):
                self._get_metric_per_sample(target, hypo)

    def reset(self):
        """
        Resets metric
        """
        self._sum_wer = 0
        self._sum_words = 0
        self._cur_wer = 0

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {"direction": "higher-worse", "type": "WER"}}

    def _get_metric_per_sample(self, annotation, prediction):
        cur_wer = self._editdistance_eval(annotation.split(), prediction.split())
        cur_words = len(annotation.split())

        self._sum_wer += cur_wer
        self._sum_words += cur_words
        self._cur_wer = cur_wer / cur_words if cur_words != 0 else 0

    def _editdistance_eval(self, source, target):
        n, m = len(source), len(target)

        distance = np.zeros((n + 1, m + 1), dtype=int)
        distance[:, 0] = np.arange(0, n + 1)
        distance[0, :] = np.arange(0, m + 1)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if source[i - 1] == target[j - 1] else 1

                distance[i][j] = min(
                    distance[i - 1][j] + 1,
                    distance[i][j - 1] + 1,
                    distance[i - 1][j - 1] + cost,
                )
        return distance[n][m]


def download_and_extract_file(url, root_path):
    os.makedirs(root_path, exist_ok=True)
    file_name = url.split("/")[-1]

    file_path = os.path.join(root_path, file_name)
    extract_path = os.path.join(root_path, file_name.split(".")[0])

    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            logger.info("Downloading %s" % file_name)
            response = requests.get(url, stream=True)
            total_length = response.headers.get("content-length")
            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (50 - done)))
                    sys.stdout.flush()

    if not os.path.exists(extract_path) and file_path.endswith("tar.gz"):
        with tarfile.open(file_path) as f:
            f.extractall(extract_path)


def main():

    # Download model
    subprocess.call(
        "omz_downloader --name mozilla-deepspeech-0.8.2 -o models", shell=True
    )

    # Convert model to IR
    subprocess.call(
        "omz_converter --name mozilla-deepspeech-0.8.2 -d models", shell=True
    )

    # Download dataset
    download_and_extract_file(
        "https://www.openslr.org/resources/12/dev-clean.tar.gz", "datasets"
    )

    # CTCDecoder for deepspeech
    decoder = SeqCtcLmDecoder(
        list(" abcdefghijklmnopqrstuvwxyz'"),
        500,
        max_candidates=1,
        scorer_lm_fname="models/public/mozilla-deepspeech-0.8.2/deepspeech-0.8.2-models.kenlm",
        alpha=0.93128901720047,
        beta=1.1834137439727783,
    )

    model_config = {
        "model_name": "mozilla-deepspeech-0.8.2",
        "model": "models/public/mozilla-deepspeech-0.8.2/FP32/mozilla-deepspeech-0.8.2.xml",
        "weights": "models/public/mozilla-deepspeech-0.8.2/FP32/mozilla-deepspeech-0.8.2.bin",
    }

    # 'recurrent_out_in_map' indicates which model output is recurrently connected to
    # which model input at next time step
    # In this case,
    # model output 'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd'
    # will be fed into model input 'previous_state_c' and
    # model output 'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd_1'
    # will be fed into model input 'previous_state_h'.
    engine_config = {
        "device": "CPU",
        "recurrent_out_in_map": {
            "cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd": "previous_state_c",
            "cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd_1": "previous_state_h",
        },
    }

    dataset_config = {
        "data_source": "./datasets",
    }

    algorithms = [
        {
            "name": "DefaultQuantization",
            "params": {
                "target_device": "ANY",
                "preset": "performance",
                "stat_subset_size": 300,
            },
        }
    ]

    # Step 1: Load the model.
    model = load_model(model_config)

    # Step 2: Initialize the data loader.
    data_loader = LibriSpeechDataLoader(dataset_config)

    # Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric.
    metric = WER(decoder)

    # Step 4: Initialize the engine for metric calculation and statistics collection.
    engine = IEEngine(config=engine_config, data_loader=data_loader, metric=metric)

    # Step 5: Create a pipeline of compression algorithms.
    pipeline = create_pipeline(algorithms, engine)

    # Step 6 (Optional): Evaluate the original model. Print the results.
    metric_results = pipeline.evaluate(model)
    if metric_results:
        print("Original model")
        for name, value in metric_results.items():
            print("{: <10s}: {}".format(name, value))

    # Step 7: Execute the pipeline.
    compressed_model = pipeline.run(model)

    # Step 8 (Optional): Compress model weights to quantized precision
    #                    in order to reduce the size of final .bin file.
    compress_model_weights(compressed_model)

    # Step 9: Save the compressed model to the desired path.
    compressed_model.save(os.path.join(os.path.curdir, "optimized"), "recurrent")

    # Step 10 (Optional): Evaluate the compressed model. Print the results.
    metric_results = pipeline.evaluate(compressed_model)
    if metric_results:
        print("Quantized model")
        for name, value in metric_results.items():
            print("{: <10s}: {}".format(name, value))


if __name__ == "__main__":
    main()
