from pydub import AudioSegment
from pydub.silence import detect_silence
import pyaudio
import warnings

def det_silence(time=500):

    warnings.filterwarnings("ignore")

    chunk = 500
    format = pyaudio.paInt16
    channels = 1
    rate = 44100
    threshold = -5  # Silence threshold in db. Depends on the environment noise. Usually should be in the range [-10,0]
    interval = []

    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    st = False
    while True:
        # This parameter indirectly determines the time the robot will wait to execute the next action
        if interval.__len__() >= time/10:
            interval.pop(0)

        data = stream.read(chunk)
        sound = AudioSegment(data, sample_width=p.get_sample_size(format), frame_rate=rate, channels=channels)
        silence = detect_silence(sound, min_silence_len=10, silence_thresh=threshold)
        if silence:
            interval.append(0)
        else:
            interval.append(1)
            st = True
        if interval.count(1) == 0 and st is True:
            break
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("success")
    return True