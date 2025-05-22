from typing import Any
import whisper


class S2T:

    def __init__(self) -> None:
        self.model = whisper.load_model("large")
        
    def __call__(self, audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio=audio, n_mels=128).to(self.model.device)
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)
        return result.text
    
s2t = S2T()
print(s2t("/workspace/ai_intern/kietpg/adapter_run-16k.wav"))