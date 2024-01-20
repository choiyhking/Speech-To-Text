import torch
import zipfile
import torchaudio
import time
from glob import glob

device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details

# download a single file in any format compatible with TorchAudio
torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',
                               dst ='speech_orig.wav', progress=True)
test_files = glob('LJSpeech-1.1/wavs/*.wav')
batches = split_into_batches(test_files, batch_size=10)

total_processing_time = 0.0
t = time.time()
for batch in batches:
    input = prepare_model_input(read_batch(batch), device=device)
    output = model(input)
    for example in output:
        print("result: ", decoder(example.cpu()))

processing_time = time.time() - t
total_processing_time += processing_time
print(f"Detection time: {processing_time:.3f}")
