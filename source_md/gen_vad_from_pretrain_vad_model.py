from pathlib import Path
from tempfile import NamedTemporaryFile

from typer.testing import CliRunner

from main import app
from vad.data_models.voice_activity import VoiceActivity

runner = CliRunner()


#def test_predict(audio_path, vad_model_path):
def test_predict():
    audio_path = "/home/maduo/codebase/fairseq_speechtext/examples/speaker_diarization/CTS-CN-F2F-2019-11-15-1287.wav"
    checkpoint_path = "/home/maduo/codebase/voice-activity-detection/tests/checkpoints/vad/sample.checkpoint"
    
    #with NamedTemporaryFile(suffix="test.json") as temp_file:
    result = runner.invoke(
        app,
        [
            "predict",
            audio_path,
            checkpoint_path,
            "--output-path",
            "test.json",
        ],
    )
    assert result.exit_code == 0
    print(f"temp_file.name: {temp_file.name}")
    voice_activity = VoiceActivity.load(Path("test.json"))
    assert len(voice_activity.activities) > 0


if __name__ == "__main__":
    test_predict()

