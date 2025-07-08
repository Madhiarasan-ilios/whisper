from flask import Flask, request, jsonify
import whisperx
import os
import tempfile

app = Flask(__name__)

@app.route("/transcribe", methods=["POST"])
def transcribe_api():
    if "audio" not in request.files or "token" not in request.form:
        return jsonify({"error": "Missing audio file or Hugging Face token"}), 400

    audio_file = request.files["audio"]
    hf_auth_token = request.form["token"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_path = temp_audio.name
        audio_file.save(audio_path)

    try:
        device = "cpu"
        batch_size = 16
        compute_type = "int8"

        # Load WhisperX model
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=batch_size)

        # Alignment
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # Diarization
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_auth_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Format final output
        def format_conversation(segments):
            formatted_output = []
            for segment in segments:
                speaker = segment.get("speaker", "Unknown Speaker")
                text = segment.get("text", "").strip()
                if speaker.startswith("SPEAKER_"):
                    try:
                        speaker_id = int(speaker.split('_')[-1])
                        speaker_label = f"Speaker {speaker_id + 1}"
                    except ValueError:
                        speaker_label = speaker
                else:
                    speaker_label = speaker
                formatted_output.append(f"{speaker_label}: {text}")
            return "\n".join(formatted_output)

        final_output = format_conversation(result["segments"])
        return final_output, 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# ⚠️ Do NOT include app.run() — Gunicorn will serve the app
