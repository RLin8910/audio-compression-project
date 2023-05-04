# audio-compression-project

## Example Usage
### Compressing and Decompressing a File
Navigate to the `src` folder, and run python from there. From there, run the following code:

```
from problems.time_domain import TimeDomain
prob = TimeDomain() # initialize the problem
prob.load_model('../examples/models/time_domain') # load the previously-trained weights
prob.compress_and_export('../examples/samp/test.wav', '../examples/samp/exp.pth') # compress the test file to 'exp.pth'
prob.decompress_and_export('../examples/samp/exp.pth', '../examples/samp/test_exp.wav') # decompress the test file to 'test_exp.wav'
```
The compressed version of the original file can be found at `exp.pth`, while the reconstructed version can be found at `test_exp.wav`.

For the frequency domain model:

```
from problems.freq_domain import FreqDomain
prob = FreqDomain() # initialize the problem
prob.load_model('../examples/models/freq_domain') # load the previously-trained weights
prob.compress_and_export('../examples/samp/test.wav', '../examples/samp/exp.pth') # compress the test file to 'exp.pth'
prob.decompress_and_export('../examples/samp/exp.pth', '../examples/samp/test_exp.wav') # decompress the test file to 'test_exp.wav'
```

### Training the Model
To train the model on data in a folder, run the following code with the model initialized:

```
from train import train
train(prob, epochs, data_root)
```

Where `epochs` should be an integer >= 1 which is the number of epochs through the data set, and `data_root` is the path to the root folder containing all training data files.

### Saving/Loading the Model
Sometimes, it can be useful to save the trained weights of the model and load them back at a later time, as we saw earlier with loading pre-trained weights in the compressing and decompressing section. To save a model from a problem `prob`:
```
prob.save_model(path)
```
To load a model from path `path`:
```
prob.load_model(path)
```
In both cases, `path` should represent a folder, not a file path, because the autoencoder will be saved as a root folder containing 2 files: one for the encoder, and another for the decoder.