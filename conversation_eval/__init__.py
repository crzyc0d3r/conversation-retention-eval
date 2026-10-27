"""Native conversational evaluation with an explicit offline replay mode."""

import os

# Keep local evaluation independent of ambient telemetry and dotenv files.
# These settings must precede the first DeepEval import.
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "YES"
os.environ["DEEPEVAL_FILE_SYSTEM"] = "READ_ONLY"
os.environ["CONFIDENT_TRACE_SAMPLE_RATE"] = "0"
os.environ["CONFIDENT_TRACE_FLUSH"] = "NO"
