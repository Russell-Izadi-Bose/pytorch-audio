#include <torch/script.h>

namespace torchaudio {

namespace {

bool is_sox_available() {
#ifdef INCLUDE_SOX
  return true;
#else
  return false;
#endif
}

bool is_kaldi_available() {
#ifdef INCLUDE_KALDI
  return true;
#else
  return false;
#endif
}

// It tells whether torchaudio was compiled with ffmpeg
// not the runtime availability.
bool is_ffmpeg_available() {
#ifdef USE_FFMPEG
  return true;
#else
  return false;
#endif
}

} // namespace

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::is_sox_available", &is_sox_available);
  m.def("torchaudio::is_kaldi_available", &is_kaldi_available);
  m.def("torchaudio::is_ffmpeg_available", &is_ffmpeg_available);
}

} // namespace torchaudio
