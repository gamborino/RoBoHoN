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
    buffer = []
    curr_max = 0

    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    while True:
        # This parameter indirectly determines the time the robot will wait to execute the next action
        if len(buffer) >= time/10:
            buffer.pop(0)
        data = stream.read(chunk)
        sound = AudioSegment(data, sample_width=p.get_sample_size(format), frame_rate=rate, channels=channels)
        silence = detect_silence(sound, min_silence_len=10, silence_thresh=threshold)
        buffer.append(0) if silence else buffer.append(1)
        # Maintain the maximum number of time intervals with sound in the buffer
        prev_max = curr_max
        curr_max = buffer.count(1)
        if curr_max < prev_max:
            curr_max = prev_max
        if buffer.count(1) == 0 and len(buffer) == time/10:
            if curr_max > 5:
                stream.stop_stream()
                stream.close()
                p.terminate()
                print("Speech over")
                return True # Speech is over
            else: 
                stream.stop_stream()
                stream.close()
                p.terminate()
                print("Silence")
                return False # No speech was detected    