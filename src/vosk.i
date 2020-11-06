#if SWIGPYTHON
%module(package="vosk", "threads"=1) vosk
#else
%module Vosk
#endif

%include <typemaps.i>

#if SWIGPYTHON
%include <pybuffer.i>
#elif SWIGJAVA
%include <various.i>
#elif SWIGCSHARP
%include <arrays_csharp.i>
#endif

#if SWIGPYTHON
%pybuffer_binary(const char *data, int len);
#endif

#if SWIGJAVA
%apply char *BYTE {const char *data};
%typemap(javaimports) KaldiRecognizer %{
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
%}
%typemap(javacode) KaldiRecognizer %{
  public boolean AcceptWaveform(byte[] data) {
    return AcceptWaveform(data, data.length);
  }
  public boolean AcceptWaveform(short[] data, int len) {
    byte[] bdata = new byte[len * 2];
    ByteBuffer.wrap(bdata).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(data, 0, len);
    return AcceptWaveform(bdata, bdata.length);
  }
%}
%pragma(java) jniclasscode=%{
    static {
        System.loadLibrary("vosk_jni");
    }
%}
#endif

#if SWIGCSHARP
CSHARP_ARRAYS(char, byte)
%apply char INPUT[] {const char *data};
%apply float INPUT[] {const float *fdata};
%apply short INPUT[] {const short *sdata};
#endif


#if SWIGJAVASCRIPT
%begin %{
#include <v8.h>
#include <node.h>
#include <node_buffer.h>
%}
#endif

%{
#include "vosk_api.h"
typedef struct VoskModel Model;
typedef struct VoskSpkModel SpkModel;
typedef struct VoskRecognizer KaldiRecognizer;
%}

typedef struct {} Model;
typedef struct {} SpkModel;
typedef struct {} KaldiRecognizer;

%extend Model {
    Model(const char *acmodel_path, const char *langmodel_path, const char *config_file_path)  {
        return vosk_model_new(acmodel_path, langmodel_path, config_file_path);
    }
    int GetSampleFrequecy(){
        return vosk_get_sample_frequency($self);
    }
    ~Model() {
        vosk_model_free($self);
    }
}

%extend SpkModel {
    SpkModel(const char *model_path)  {
        return vosk_spk_model_new(model_path);
    }
    ~SpkModel() {
        vosk_spk_model_free($self);
    }
}

%extend KaldiRecognizer {
    KaldiRecognizer(Model *model, SpkModel *spk_model, float sample_rate, bool online)  {
        return vosk_recognizer_new(model, spk_model, sample_rate, online);
    }
    KaldiRecognizer(Model *model, float sample_rate, const char* grammar, bool online)  {
        return vosk_recognizer_new_grm(model, sample_rate, grammar, online);
    }
    ~KaldiRecognizer() {
        vosk_recognizer_free($self);
    }

#if SWIGCSHARP
    bool AcceptWaveform(const char *data, int len) {
        return vosk_recognizer_accept_waveform($self, data, len);
    }
    bool AcceptWaveform(const short *sdata, int len) {
        return vosk_recognizer_accept_waveform_s($self, sdata, len);
    }
    bool AcceptWaveform(const float *fdata, int len) {
        return vosk_recognizer_accept_waveform_f($self, fdata, len);
    }
#elif SWIGJAVA
    bool AcceptWaveform(const char *data, int len) {
        return vosk_recognizer_accept_waveform($self, data, len);
    }
#elif SWIGJAVASCRIPT
    bool AcceptWaveform(SWIG_Object ptr) {
        char* data = (char*) node::Buffer::Data(ptr);
        size_t length = node::Buffer::Length(ptr);
        return vosk_recognizer_accept_waveform($self, data, length);
    }
#else
    int AcceptWaveform(const char *data, int len) {
        return vosk_recognizer_accept_waveform($self, data, len);
    }
#endif

    const char* Result() {
        return vosk_recognizer_result($self);
    }
    const char* PartialResult() {
        return vosk_recognizer_partial_result($self);
    }
    const char* FinalResult() {
        return vosk_recognizer_final_result($self);
    }
    const char* GetMetadata() {
        return vosk_recognizer_get_metadata($self);
    }

    float uttConfidence() {
        return vosk_recognizer_uttConfidence($self);
    }
}

%rename(SetLogLevel) vosk_set_log_level;
void vosk_set_log_level(int level);
