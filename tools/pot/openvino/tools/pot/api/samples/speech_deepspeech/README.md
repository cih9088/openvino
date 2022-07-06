# Quantizing Recurrent Model {#pot_example_deepspeech_README}

This example demonstrates the use of the [Post-training Optimization Tool API](@ref pot_compression_api_README) for the task of quantizing a automatic speech recognition (ASR) model.
The [DeepSpeech](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/mozilla-deepspeech-0.8.2/README.md) model from Tensorflow is used for thie purpose.
A Custom `DataLoader` is created to load [LibriSpeech](https://www.openslr.org/12) dataset for a ASR task and the implementation of Word Error Rate (WER) metric is used for the model evaluation. In addition, this example demonstrates how one use `recurrent_out_in_map` and `inference_fn` to infer a recurrent model. The code of the example is available on [GitHub](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/speech_deepspeech).

## How to run the example
1. Install [ctcdecode-numpy](https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/speech_recognition_deepspeech_demo/python/ctcdecode-numpy) from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) demo for CTC beam decoding.
   ```sh
   git clone https://github.com/openvinotoolkit/open_model_zoo.git
   python -m pip install open_model_zoo/demos/speech_recognition_deepspeech_demo/python/ctcdecode-numpy
   ```

2. Luanche the example script from the example directory:
   ```sh
   python deepspeech_recurrent_sample.py
   ```
   ```sh
   python deepspeech_custom_inference_sample.py
   ```
