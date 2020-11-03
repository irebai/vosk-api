// Copyright 2019 Alpha Cephei Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


//
// For details of possible model layout see doc/models.md section model-structure


// Modifications are apported by Ilyes Rebai, research engineer at Linagora.
// The main goal is to adapt the model files' loading w.r.t. Linagora Models' configuration
// and to adapt the code to the original KALDI decoder one "online2-wav-nnet3-latgen-faster".
// Use extrat variables to make the decoder parameters more flexible.
// Contact: irebai@linagora.com


#include "model.h"

#include <sys/stat.h>
#include <fst/fst.h>
#include <fst/register.h>
#include <fst/matcher-fst.h>
#include <fst/extensions/ngram/ngram-fst.h>

namespace fst {

static FstRegisterer<StdOLabelLookAheadFst> OLabelLookAheadFst_StdArc_registerer;
static FstRegisterer<NGramFst<StdArc>> NGramFst_StdArc_registerer;

}  // namespace fst

#ifdef __ANDROID__
#include <android/log.h>
static void KaldiLogHandler(const LogMessageEnvelope &env, const char *message)
{
  int priority;
  if (env.severity > GetVerboseLevel())
      return;

  if (env.severity > LogMessageEnvelope::kInfo) {
    priority = ANDROID_LOG_VERBOSE;
  } else {
    switch (env.severity) {
    case LogMessageEnvelope::kInfo:
      priority = ANDROID_LOG_INFO;
      break;
    case LogMessageEnvelope::kWarning:
      priority = ANDROID_LOG_WARN;
      break;
    case LogMessageEnvelope::kAssertFailed:
      priority = ANDROID_LOG_FATAL;
      break;
    case LogMessageEnvelope::kError:
    default: // If not the ERROR, it still an error!
      priority = ANDROID_LOG_ERROR;
      break;
    }
  }

  std::stringstream full_message;
  full_message << env.func << "():" << env.file << ':'
               << env.line << ") " << message;

  __android_log_print(priority, "VoskAPI", "%s", full_message.str().c_str());
}
#else
static void KaldiLogHandler(const LogMessageEnvelope &env, const char *message)
{
  if (env.severity > GetVerboseLevel())
      return;

  // Modified default Kaldi logging so we can disable LOG messages.
  std::stringstream full_message;
  if (env.severity > LogMessageEnvelope::kInfo) {
    full_message << "VLOG[" << env.severity << "] (";
  } else {
    switch (env.severity) {
    case LogMessageEnvelope::kInfo:
      full_message << "LOG (";
      break;
    case LogMessageEnvelope::kWarning:
      full_message << "WARNING (";
      break;
    case LogMessageEnvelope::kAssertFailed:
      full_message << "ASSERTION_FAILED (";
      break;
    case LogMessageEnvelope::kError:
    default: // If not the ERROR, it still an error!
      full_message << "ERROR (";
      break;
    }
  }
  // Add other info from the envelope and the message text.
  full_message << "VoskAPI" << ':'
               << env.func << "():" << env.file << ':'
               << env.line << ") " << message;

  // Print the complete message to stderr.
  full_message << "\n";
  std::cerr << full_message.str();
}
#endif

Model::Model(const char *acmodel_path, const char *langmodel_path, const char *config_file_path) : acmodel_path_str_(acmodel_path), langmodel_path_str_(langmodel_path), config_file_path_str_(config_file_path) {

    SetLogHandler(KaldiLogHandler);
    Configure();
    ReadDataFiles();

    ref_cnt_ = 1;
}

void Model::Configure()
{
    struct stat buffer;

    kaldi::ParseOptions po("something");
    nnet3_decoding_config_.Register(&po);
    endpoint_config_.Register(&po);
    decodable_opts_.Register(&po);
    feature_config_.Register(&po);

    if (stat(config_file_path_str_.c_str(), &buffer) == 0){
      KALDI_LOG << "Loading decode config file from " << config_file_path_str_;
      po.ReadConfigFile(config_file_path_str_);
    } else {
      po.ReadConfigFile(acmodel_path_str_ + "/conf/online.conf"); }


    nnet3_rxfilename_ = acmodel_path_str_ + "/final.mdl";
    hclg_fst_rxfilename_ = langmodel_path_str_ + "/HCLG.fst";
    hcl_fst_rxfilename_ = langmodel_path_str_ + "/HCLr.fst";
    g_fst_rxfilename_ = langmodel_path_str_ + "/Gr.fst";
    disambig_rxfilename_ = langmodel_path_str_ + "/disambig_tid.int";
    word_syms_rxfilename_ = langmodel_path_str_ + "/words.txt";
    winfo_rxfilename_ = langmodel_path_str_ + "/word_boundary.int";
    carpa_rxfilename_ = langmodel_path_str_ + "/rescore/G.carpa";
    std_fst_rxfilename_ = langmodel_path_str_ + "/rescore/G.fst";
}

void Model::ReadDataFiles()
{
    struct stat buffer;

    //load feature extraction config and ivector extractiong config
    feature_info_ = new kaldi::OnlineNnet2FeaturePipelineInfo (feature_config_);

    //load acoustic model and decode config
    KALDI_LOG << "Am model file "<< nnet3_rxfilename_;
    trans_model_ = new kaldi::TransitionModel();
    nnet_ = new kaldi::nnet3::AmNnetSimple();
    {
        bool binary;
        kaldi::Input ki(nnet3_rxfilename_, &binary);
        trans_model_->Read(ki.Stream(), binary);
        nnet_->Read(ki.Stream(), binary);
        SetBatchnormTestMode(true, &(nnet_->GetNnet()));
        SetDropoutTestMode(true, &(nnet_->GetNnet()));
        nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(nnet_->GetNnet()));
    }
    decodable_info_ = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts_,
                                                               nnet_);

    //load decode graph
    if (stat(hclg_fst_rxfilename_.c_str(), &buffer) == 0) {
        KALDI_LOG << "Loading HCLG from " << hclg_fst_rxfilename_;
        hclg_fst_ = fst::ReadFstKaldiGeneric(hclg_fst_rxfilename_);
        hcl_fst_ = NULL;
        g_fst_ = NULL;
    } else {
        KALDI_LOG << "Loading HCL and G from " << hcl_fst_rxfilename_ << " " << g_fst_rxfilename_;
        hclg_fst_ = NULL;
        hcl_fst_ = fst::StdFst::Read(hcl_fst_rxfilename_);
        g_fst_ = fst::StdFst::Read(g_fst_rxfilename_);
        ReadIntegerVectorSimple(disambig_rxfilename_, &disambig_);
    }

    //load word symbol
    word_syms_ = NULL;
    if (hclg_fst_ && hclg_fst_->OutputSymbols()) {
        word_syms_ = hclg_fst_->OutputSymbols();
    } else if (g_fst_ && g_fst_->OutputSymbols()) {
        word_syms_ = g_fst_->OutputSymbols();
    }
    if (!word_syms_) {
        KALDI_LOG << "Loading words from " << word_syms_rxfilename_;
        if (!(word_syms_ = fst::SymbolTable::ReadText(word_syms_rxfilename_)))
            KALDI_ERR << "Could not read symbol table from file "
                      << word_syms_rxfilename_;
    }
    KALDI_ASSERT(word_syms_);

    //load word boundary used to compute word timestamps
    if (stat(winfo_rxfilename_.c_str(), &buffer) == 0) {
        KALDI_LOG << "Loading winfo " << winfo_rxfilename_;
        kaldi::WordBoundaryInfoNewOpts opts;
        winfo_ = new kaldi::WordBoundaryInfo(opts, winfo_rxfilename_);
    } else {
        winfo_ = NULL;
    }

    //load rescoring graphs
    if (stat(carpa_rxfilename_.c_str(), &buffer) == 0) {
        KALDI_LOG << "Loading CARPA model from " << carpa_rxfilename_;
        std_lm_fst_ = fst::ReadFstKaldi(std_fst_rxfilename_);
        fst::Project(std_lm_fst_, fst::PROJECT_OUTPUT);
        if (std_lm_fst_->Properties(fst::kILabelSorted, true) == 0) {
            fst::ILabelCompare<fst::StdArc> ilabel_comp;
            fst::ArcSort(std_lm_fst_, ilabel_comp);
        }
        ReadKaldiObject(carpa_rxfilename_, &const_arpa_);
    } else {
        std_lm_fst_ = NULL;
    }

    //load cmvn matrix used during ivector extraction
    if (feature_config_.global_cmvn_stats_rxfilename != "")
    {
        KALDI_LOG << "Loading global CMVN stats from " << feature_config_.global_cmvn_stats_rxfilename;
        ReadKaldiObject(feature_config_.global_cmvn_stats_rxfilename,
                      &global_cmvn_stats_);
    }
    cmvn_state_ = new kaldi::OnlineCmvnState (global_cmvn_stats_);

    // activate wave upsample/downsample
    if (feature_config_.feature_type == "mfcc") {
      feature_info_->mfcc_opts.frame_opts.allow_downsample = true; // It is safe to downsample
      feature_info_->mfcc_opts.frame_opts.allow_upsample = true; // It is safe to upsample
    } else if (feature_config_.feature_type == "plp") {
      feature_info_->plp_opts.frame_opts.allow_downsample = true; // It is safe to downsample
      feature_info_->plp_opts.frame_opts.allow_upsample = true; // It is safe to upsample
    } else if (feature_config_.feature_type == "fbank") {
      feature_info_->fbank_opts.frame_opts.allow_downsample = true; // It is safe to downsample
      feature_info_->fbank_opts.frame_opts.allow_upsample = true; // It is safe to upsample
    } else {
      KALDI_ERR << "Code error: invalid feature type " << feature_config_.feature_type;
    }

    //set silence phones for ivector update and create ivector adaptation state object
    feature_info_->silence_weighting_config.silence_phones_str = endpoint_config_.silence_phones;
    adaptation_state_ = new kaldi::OnlineIvectorExtractorAdaptationState(feature_info_->ivector_extractor_info);


    //save the default sample frequence
    sample_frequence_ = feature_info_->mfcc_opts.frame_opts.samp_freq;

    //Check silence_weighting is activeted or not
    string silweightstatus = (!feature_info_->silence_weighting_config.silence_phones_str.empty() &&
      feature_info_->silence_weighting_config.silence_weight != 1.0) ? "activated (weight=" + to_string(feature_info_->silence_weighting_config.silence_weight) + ")" : "deactivated";
    KALDI_LOG << "Ivector silence weighting is " << silweightstatus;

}

void Model::Debug()
{
    KALDI_LOG << "Decoding params beam=" << nnet3_decoding_config_.beam <<
         " max-active=" << nnet3_decoding_config_.max_active <<
         " lattice-beam=" << nnet3_decoding_config_.lattice_beam;
    KALDI_LOG << "Silence phones " << endpoint_config_.silence_phones;
    KALDI_LOG << "feature type " << feature_config_.feature_type;
    KALDI_LOG << feature_info_->ivector_extractor_info.ivector_period;
    KALDI_LOG << feature_info_->ivector_extractor_info.greedy_ivector_extractor;
    KALDI_LOG << feature_info_->ivector_extractor_info.max_count;
    KALDI_LOG << feature_info_->ivector_extractor_info.max_remembered_frames;
    KALDI_LOG << feature_info_->ivector_extractor_info.min_post;
    KALDI_LOG << feature_info_->ivector_extractor_info.num_cg_iters;
    KALDI_LOG << feature_info_->ivector_extractor_info.num_gselect;
    KALDI_LOG << feature_info_->ivector_extractor_info.posterior_scale;
    KALDI_LOG << feature_info_->ivector_extractor_info.use_most_recent_ivector;
    KALDI_LOG << feature_info_->ivector_extractor_info.lda_mat;
    KALDI_LOG << feature_info_->ivector_extractor_info.splice_opts.left_context;
    KALDI_LOG << feature_info_->ivector_extractor_info.splice_opts.right_context;
    KALDI_LOG << feature_info_->silence_weighting_config.silence_weight;
    KALDI_LOG << feature_info_->silence_weighting_config.silence_phones_str;
    KALDI_LOG << decodable_opts_.extra_left_context_initial;
    KALDI_LOG << decodable_opts_.frames_per_chunk;
}

int Model::getSampleFreq()
{
  return sample_frequence_;
}

void Model::Ref() 
{
    ref_cnt_++;
}

void Model::Unref() 
{
    ref_cnt_--;
    if (ref_cnt_ == 0) {
        delete this;
    }
}

Model::~Model() {
    delete decodable_info_;
    delete trans_model_;
    delete nnet_;
    delete winfo_;
    delete hclg_fst_;
    delete hcl_fst_;
    delete g_fst_;
}
