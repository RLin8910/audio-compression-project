import numpy
import wave

"""
Open a wave file at the path `path` and return it as a dictionary containing
frames as a numpy array as well as other characteristics.
"""
def import_to_array(path):
    file = wave.open(path)
    sample_count = file.getnframes()
    frames = file.readframes(sample_count)

    audio_int = numpy.frombuffer(frames, dtype=numpy.int16)
    audio_float = audio_int.astype(numpy.float32)

    result = {}
    result['framerate'] = file.getframerate()
    result['channels'] = file.getnchannels()
    result['sampwidth'] = file.getsampwidth()
    result['frames'] = audio_float

    return result

"""
Export a sound to a wave file at the path `path`.
"""
def export_to_file(frames, framerate, channels, sampwidth, path):
    with wave.open(path, "w") as file:
        frames_int = frames.astype(numpy.int16)

        file.setnchannels(channels)
        file.setsampwidth(sampwidth)
        file.setframerate(framerate)
        file.writeframes(frames_int.tobytes())