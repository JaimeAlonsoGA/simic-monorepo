"""Shared constants — values used across multiple modules."""

# Speaker labels
SPEAKER_YOU = "You"
SPEAKER_OTHER = "Other"
SPEAKER_LABEL_FMT = "Speaker {}"

# Audio file names (within output/temp directories)
FILE_STEREO = "stereo.wav"
FILE_MONO_16K = "mono_16k.wav"
FILE_MIC_16K = "mic_16k.wav"
FILE_MONITOR_16K = "monitor_16k.wav"
# A session records one file per channel, named after it, with the capture format's
# extension: monitor.mp3, mic.mp3.
CHANNEL_MONITOR = "monitor"
CHANNEL_MIC = "mic"
FILE_SESSION = "session.json"
# Held (flock) by the process turning a session directory into a note.
FILE_PROCESSING_LOCK = ".processing.lock"
# Held (flock), in the state directory, by whichever process is transcribing: one at a time.
FILE_TRANSCRIBE_LOCK = ".transcribe.lock"
# What a session processed in the background printed, kept in its directory.
FILE_PROCESSING_LOG = "processing.log"

# The desktop app's id: the window registers under it, and its launcher is named after
# it, which is how the desktop matches the two.
APP_ID = "io.github.jaimealonsoga.simic"

# Where in-progress recordings live is not a constant: it depends on settings and
# XDG_STATE_HOME, and is verified to be private before use. See recorder.session_root().

# Disk one hour of meeting needs until its note exists: two 64 kbps MP3 channels plus
# the 16 kHz stereo and per-channel copies processing writes (measured: 518 MB).
BYTES_PER_MEETING_HOUR = 520_000_000
# Below this much room, `start` and `doctor` warn: meetings run for hours.
MIN_FREE_MEETING_HOURS = 2
BYTES_PER_GB = 1_000_000_000

# Sample rates
SAMPLE_RATE_16K = 16000

# Remote STT upload limits (OpenAI documents a 25 MiB hard cap; stay under it).
# Bitrate chosen so ~50 minutes of mono speech fits in one upload at this margin.
STT_REMOTE_MAX_UPLOAD_BYTES = 24 * 1024 * 1024
STT_REMOTE_MP3_BITRATE_K = 64
# Leave headroom under the hard upload cap when sizing time slices from bitrate.
STT_REMOTE_UPLOAD_MARGIN = 0.95
# Truncate ffmpeg stderr in errors so logs stay readable.
STT_REMOTE_FFMPEG_ERROR_TAIL = 500
# OpenAI gpt-4o-transcribe-diarize requires chunking_strategy above this duration.
OPENAI_DIARIZE_CHUNKING_SECONDS = 30.0
# Hard API duration caps (size limit alone is not enough for these models).
OPENAI_DIARIZE_MAX_UPLOAD_SECONDS = 1400.0
OPENAI_GPT_TRANSCRIBE_MAX_UPLOAD_SECONDS = 1500.0
# Soft targets keep each upload well under the hard cap and under one read-timeout
# budget (near-max diarize slices timed out after ~30 min of SDK retries).
OPENAI_DIARIZE_TARGET_UPLOAD_SECONDS = 600.0
OPENAI_GPT_TRANSCRIBE_TARGET_UPLOAD_SECONDS = 600.0
# Connect timeout for the shared OpenAI client (read/write come from settings).
STT_REMOTE_CONNECT_TIMEOUT = 10.0
# Cap exponential backoff between remote STT upload attempts.
STT_REMOTE_RETRY_DELAY_CAP = 60.0

# Frames read per iteration when de-interleaving a stereo WAV. Large enough that the
# per-chunk overhead is irrelevant, small enough that the transient buffer (~4 MB at
# this size) does not matter next to the output arrays.
READ_CHUNK_FRAMES = 1_000_000

# Audio channel layout
STEREO_CHANNELS = 2
# Minimum kept sub-segment duration when splitting on silence without word timings
MIN_SUB_SEGMENT_DURATION_SEC = 0.5

# FFmpeg loudnorm parameters (EBU R128)
LOUDNORM_PARAMS = "I=-16:TP=-1.5:LRA=11"

# Channel energy classification
CHANNEL_ENERGY_RATIO = 2.0
CHANNEL_EPSILON = 1e-10

# Silence detection
SILENCE_WINDOW_SEC = 0.1
SILENCE_ADAPTIVE_FACTOR = 0.4
SILENCE_MONITOR_FACTOR = 0.3
# RMS energy floor (raw int16 scale) below which audio counts as silence
SILENCE_RMS_THRESHOLD = 200.0

# Spectral analysis
SPECTRAL_FFT_SIZE = 2048
SPECTRAL_MIN_FREQ_HZ = 100.0
SPECTRAL_MAX_FREQ_HZ = 4000.0

# Pyannote
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

# PulseAudio/PipeWire
PA_DEFAULT_MONITOR = "@DEFAULT_MONITOR@"
PA_DEFAULT_SOURCE = "@DEFAULT_SOURCE@"
PA_MONITOR_SUFFIX = ".monitor"

# Duration warning threshold (seconds)
CHANNEL_DURATION_DIFF_WARN = 2.0

# Minimum segment duration for output (seconds)
MIN_SEGMENT_DURATION = 1.0

# Phrases Whisper emits from its subtitle training corpus rather than from the audio,
# typically over long pauses. Matched case-insensitively as substrings. Kept here
# because both the quality benchmark and the transcript filter must agree on the list.
HALLUCINATION_MARKERS = (
    "субтитры",
    "dimatorzok",
    "продолжение следует",
    "редактор субтитров",
    "корректор",
    "субтитлы",
    "amara.org",
    "субтитри",
    "thanks for watching",
    "thank you for watching",
    "подписывайтесь на канал",
)

# Live transcription
FILE_LIVE_SUFFIX = "_live"
WAV_HEADER_FALLBACK = 44
WAV_CHUNK_HEADER_BYTES = 4  # RIFF chunk id / size fields are 4 bytes each

# API base URLs for LLM providers
API_BASE_GROQ = "https://api.groq.com/openai/v1"
API_BASE_GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_BASE_OPENROUTER = "https://openrouter.ai/api/v1"
API_BASE_DEEPSEEK = "https://api.deepseek.com"
API_BASE_QWEN = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
