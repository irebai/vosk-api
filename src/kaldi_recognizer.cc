// Copyright 2019-2020 Alpha Cephei Inc.
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


// Modifications are apported by Ilyes Rebai, research engineer at Linagora.
// The main goal is to adapt the code to the original KALDI decoder one "online2-wav-nnet3-latgen-faster".
// Offline/Online mode is now supported.
// Metadata is separated from the result.
// Out-of-memory error is hundled.
// Contact: irebai@linagora.com


#include "kaldi_recognizer.h"
#include "fstext/fstext-utils.h"
#include "lat/sausages.h"

using namespace fst;
using namespace kaldi::nnet3;

KaldiRecognizer::KaldiRecognizer(Model *model, SpkModel *spk_model, float sample_frequency, bool online) : model_(model), spk_model_(spk_model), sample_frequency_(sample_frequency), online_(online) {

    model_->Ref();

    if (online_) {
        model_->feature_info_->ivector_extractor_info.use_most_recent_ivector = true;
        model_->feature_info_->ivector_extractor_info.greedy_ivector_extractor = true;
    }

    feature_pipeline_ = new kaldi::OnlineNnet2FeaturePipeline (*model_->feature_info_);
    silence_weighting_ = new kaldi::OnlineSilenceWeighting(*model_->trans_model_, model_->feature_info_->silence_weighting_config, 3);

    feature_pipeline_->SetAdaptationState(*model_->adaptation_state_);
    feature_pipeline_->SetCmvnState(*model_->cmvn_state_);

    g_fst_ = NULL;
    decode_fst_ = NULL;

    if (!model_->hclg_fst_) {
        if (model_->hcl_fst_ && model_->g_fst_) {
            decode_fst_ = LookaheadComposeFst(*model_->hcl_fst_, *model_->g_fst_, model_->disambig_);
        } else {
            KALDI_ERR << "Can't create decoding graph";
        }
    }

    decoder_ = new kaldi::SingleUtteranceNnet3Decoder(model_->nnet3_decoding_config_,
            *model_->trans_model_,
            *model_->decodable_info_,
            model_->hclg_fst_ ? *model_->hclg_fst_ : *decode_fst_,
            feature_pipeline_);


    if (spk_model_)
        spk_feature_ = new OnlineMfcc(spk_model_->spkvector_mfcc_opts);
    else
        spk_feature_ = NULL;


    InitState();
    InitRescoring();
}

KaldiRecognizer::KaldiRecognizer(Model *model, float sample_frequency, char const *grammar, bool online) : model_(model), spk_model_(0), sample_frequency_(sample_frequency), online_(online){
    
    model_->Ref();

    if (online_) {
        model_->feature_info_->ivector_extractor_info.use_most_recent_ivector = true;
        model_->feature_info_->ivector_extractor_info.greedy_ivector_extractor = true;
    }

    feature_pipeline_ = new kaldi::OnlineNnet2FeaturePipeline (*model_->feature_info_);
    silence_weighting_ = new kaldi::OnlineSilenceWeighting(*model_->trans_model_, model_->feature_info_->silence_weighting_config, 3);

    feature_pipeline_->SetAdaptationState(*model_->adaptation_state_);
    feature_pipeline_->SetCmvnState(*model_->cmvn_state_);

    g_fst_ = new StdVectorFst();
    if (model_->hcl_fst_) {
        g_fst_->AddState();
        g_fst_->SetStart(0);
        g_fst_->AddState();
        g_fst_->SetFinal(1, fst::TropicalWeight::One());
        g_fst_->AddArc(1, StdArc(0, 0, fst::TropicalWeight::One(), 0));

        // Create simple word loop FST
        stringstream ss(grammar);
        string token;

        while (getline(ss, token, ' ')) {
            int32 id = model_->word_syms_->Find(token);
            if (id == kNoSymbol) {
                KALDI_WARN << "Ignoring word missing in vocabulary: '" << token << "'";
            } else {
                g_fst_->AddArc(0, StdArc(id, id, fst::TropicalWeight::One(), 1));
            }
        }
        ArcSort(g_fst_, ILabelCompare<StdArc>());

        decode_fst_ = LookaheadComposeFst(*model_->hcl_fst_, *g_fst_, model_->disambig_);
    } else {
        decode_fst_ = NULL;
        KALDI_ERR << "Can't create decoding graph";
    }

    decoder_ = new kaldi::SingleUtteranceNnet3Decoder(model_->nnet3_decoding_config_,
            *model_->trans_model_,
            *model_->decodable_info_,
            model_->hclg_fst_ ? *model_->hclg_fst_ : *decode_fst_,
            feature_pipeline_);

    spk_feature_ = NULL;

    InitState();
    InitRescoring();
}

KaldiRecognizer::~KaldiRecognizer() {
    delete decoder_;
    delete feature_pipeline_;
    delete silence_weighting_;
    delete g_fst_;
    delete decode_fst_;
    delete spk_feature_;
    delete lm_fst_;

    decoder_ = NULL;
    feature_pipeline_ = NULL;
    silence_weighting_ = NULL;
    g_fst_ = NULL;
    decode_fst_ = NULL;
    spk_feature_ = NULL;
    metadata_ = NULL;

    model_->Unref();
    if (spk_model_)
         spk_model_->Unref();
}

void KaldiRecognizer::InitState()
{
    frame_offset_ = 0;
    samples_processed_ = 0;
    samples_round_start_ = 0;

    state_ = RECOGNIZER_INITIALIZED;
}

void KaldiRecognizer::InitRescoring()
{
    if (model_->std_lm_fst_) {
        fst::CacheOptions cache_opts(true, 50000);
        fst::MapFstOptions mapfst_opts(cache_opts);
        fst::StdToLatticeMapper<kaldi::BaseFloat> mapper;
        lm_fst_ = new fst::MapFst<fst::StdArc, kaldi::LatticeArc, fst::StdToLatticeMapper<kaldi::BaseFloat> >(*model_->std_lm_fst_, mapper, mapfst_opts);
    } else {
        lm_fst_ = NULL;
    }
}

void KaldiRecognizer::CleanUp()
{
    delete silence_weighting_;
    silence_weighting_ = new kaldi::OnlineSilenceWeighting(*model_->trans_model_, model_->feature_info_->silence_weighting_config, 3);

    if (spk_model_) {
        delete spk_feature_;
        spk_feature_ = new OnlineMfcc(spk_model_->spkvector_mfcc_opts);
    }

    if (decoder_)
       frame_offset_ += decoder_->NumFramesDecoded();

    // Each 10 minutes we drop the pipeline to save frontend memory in continuous processing
    // here we drop few frames remaining in the feature pipeline but hope it will not
    // cause a huge accuracy drop since it happens not very frequently.

    // Also restart if we retrieved final result already

    if (decoder_ == NULL || state_ == RECOGNIZER_FINALIZED || frame_offset_ > 20000) {
        samples_round_start_ += samples_processed_;
        samples_processed_ = 0;
        frame_offset_ = 0;

        delete decoder_;
        delete feature_pipeline_;

        feature_pipeline_ = new kaldi::OnlineNnet2FeaturePipeline (*model_->feature_info_); 
        feature_pipeline_->SetAdaptationState(*model_->adaptation_state_);
        feature_pipeline_->SetCmvnState(*model_->cmvn_state_);
        decoder_ = new kaldi::SingleUtteranceNnet3Decoder(model_->nnet3_decoding_config_,
            *model_->trans_model_,
            *model_->decodable_info_,
            model_->hclg_fst_ ? *model_->hclg_fst_ : *decode_fst_,
            feature_pipeline_);
    } else {
        decoder_->InitDecoding(frame_offset_);
    }
}

void KaldiRecognizer::getFeatureFrames()
{
    int num_frames = feature_pipeline_->NumFramesReady();
    int dim = feature_pipeline_->Dim();

    if (model_->feature_info_->use_ivectors) {
        dim = model_->feature_info_->mfcc_opts.num_ceps;
    }

    for (int i = 0; i < num_frames; ++i) {
        json::JSON frame;
        Vector<BaseFloat> feat(feature_pipeline_->Dim());
        feature_pipeline_->GetFrame(i, &feat);
        for (int j = 0; j < dim; j++) {
            frame.append(feat(j));
        }
        metadata_["features"].append(frame);
    }
    metadata_["segments"] = silence_pos;
}

void KaldiRecognizer::UpdateSilenceWeights()
{
    if (silence_weighting_->Active() && feature_pipeline_->NumFramesReady() > 0 &&
        feature_pipeline_->IvectorFeature() != NULL) {
        vector<pair<int32, BaseFloat> > delta_weights;
        silence_weighting_->ComputeCurrentTraceback(decoder_->Decoder());
        silence_weighting_->GetDeltaWeights(feature_pipeline_->NumFramesReady(),
                                          frame_offset_ * 3,
                                          &delta_weights);
        feature_pipeline_->UpdateFrameWeights(delta_weights);
    }
}

bool KaldiRecognizer::AcceptWaveform(const char *data, int len)
{
    Vector<BaseFloat> wave;
    wave.Resize(len / 2, kUndefined);
    for (int i = 0; i < len / 2; i++)
        wave(i) = *(((short *)data) + i);
    return AcceptWaveform(wave);
}

bool KaldiRecognizer::AcceptWaveform(const short *sdata, int len)
{
    Vector<BaseFloat> wave;
    wave.Resize(len, kUndefined);
    for (int i = 0; i < len; i++)
        wave(i) = sdata[i];
    return AcceptWaveform(wave);
}

bool KaldiRecognizer::AcceptWaveform(const float *fdata, int len)
{
    Vector<BaseFloat> wave;
    wave.Resize(len, kUndefined);
    for (int i = 0; i < len; i++)
        wave(i) = fdata[i];
    return AcceptWaveform(wave);
}

bool KaldiRecognizer::AcceptWaveform(Vector<BaseFloat> &wdata)
{
    // Cleanup if we finalized previous utterance or the whole feature pipeline
    if (!(state_ == RECOGNIZER_RUNNING || state_ == RECOGNIZER_INITIALIZED)) {
        CleanUp();
    }
    state_ = RECOGNIZER_RUNNING;

    // Compute acoustic and ivector features
    feature_pipeline_->AcceptWaveform(sample_frequency_, wdata);

    if (online_) {
        // Update ivector features using computed delta weights if silence weighting is activated
        UpdateSilenceWeights();
        // Perform decoding
        decoder_->AdvanceDecoding();
    }
    
    if (spk_feature_) {
        spk_feature_->AcceptWaveform(sample_frequency_, wdata);
    }

    if (decoder_->EndpointDetected(model_->endpoint_config_)) {
        silence_pos.append(feature_pipeline_->NumFramesReady());
        return true;
    }

    samples_processed_ += wdata.Dim();

    return false;
}

// Computes an xvector from a chunk of speech features.
static void RunNnetComputation(const MatrixBase<BaseFloat> &features,
    const nnet3::Nnet &nnet, nnet3::CachingOptimizingCompiler *compiler,
    Vector<BaseFloat> *xvector) 
{
    nnet3::ComputationRequest request;
    request.need_model_derivative = false;
    request.store_component_stats = false;
    request.inputs.push_back(
    nnet3::IoSpecification("input", 0, features.NumRows()));
    nnet3::IoSpecification output_spec;
    output_spec.name = "output";
    output_spec.has_deriv = false;
    output_spec.indexes.resize(1);
    request.outputs.resize(1);
    request.outputs[0].Swap(&output_spec);
    shared_ptr<const nnet3::NnetComputation> computation = compiler->Compile(request);
    nnet3::Nnet *nnet_to_update = NULL;  // we're not doing any update.
    nnet3::NnetComputer computer(nnet3::NnetComputeOptions(), *computation,
                    nnet, nnet_to_update);
    CuMatrix<BaseFloat> input_feats_cu(features);
    computer.AcceptInput("input", &input_feats_cu);
    computer.Run();
    CuMatrix<BaseFloat> cu_output;
    computer.GetOutputDestructive("output", &cu_output);
    xvector->Resize(cu_output.NumCols());
    xvector->CopyFromVec(cu_output.Row(0));
}

#define MIN_SPK_FEATS 30

bool KaldiRecognizer::GetSpkVector(Vector<BaseFloat> &xvector)
{
    vector<int32> nonsilence_frames;
    if (silence_weighting_->Active() && feature_pipeline_->NumFramesReady() > 0) {
        silence_weighting_->ComputeCurrentTraceback(decoder_->Decoder(), true);
        silence_weighting_->GetNonsilenceFrames(feature_pipeline_->NumFramesReady(),
                                          frame_offset_ * 3,
                                          &nonsilence_frames);
    }

    int num_frames = spk_feature_->NumFramesReady();
    Matrix<BaseFloat> mfcc(num_frames, spk_feature_->Dim());

    // Not very efficient, would be nice to have faster search
    int num_nonsilence_frames = 0;
    for (int i = 0; i < num_frames; ++i) {
       if (std::find(nonsilence_frames.begin(),
                     nonsilence_frames.end(), i / 3) == nonsilence_frames.end()) {
           continue;
       }
       Vector<BaseFloat> feat(spk_feature_->Dim());
       spk_feature_->GetFrame(i, &feat);
       mfcc.CopyRowFromVec(feat, num_nonsilence_frames);
       num_nonsilence_frames++;
    }

    // Don't extract vector if not enough data
    if (num_nonsilence_frames < MIN_SPK_FEATS)
        return false;

    mfcc.Resize(num_nonsilence_frames, spk_feature_->Dim());

    SlidingWindowCmnOptions cmvn_opts;
    Matrix<BaseFloat> features(mfcc.NumRows(), mfcc.NumCols(), kUndefined);
    SlidingWindowCmn(cmvn_opts, mfcc, &features);

    nnet3::NnetSimpleComputationOptions opts;
    nnet3::CachingOptimizingCompilerOptions compiler_config;
    nnet3::CachingOptimizingCompiler compiler(spk_model_->speaker_nnet, opts.optimize_config, compiler_config);

    RunNnetComputation(features, spk_model_->speaker_nnet, &compiler, &xvector);
    return true;
}

void KaldiRecognizer::ComputeTimestamp(kaldi::CompactLattice clat)
{
    // TODO: check if it is a command ASR or LVCSR
    
    try
    {
        //fst::ScaleLattice(fst::GraphLatticeScale(0.9), &clat); // Apply rescoring weight
        CompactLattice aligned_lat;
        if (model_->winfo_) {
            WordAlignLattice(clat, *model_->trans_model_, *model_->winfo_, 0, &aligned_lat);
        } else {
            aligned_lat = clat;
        }

        MinimumBayesRisk mbr(aligned_lat);
        const vector<BaseFloat> &conf = mbr.GetOneBestConfidences();
        const vector<int32> &words = mbr.GetOneBest();
        const vector<pair<BaseFloat, BaseFloat> > &times =
            mbr.GetOneBestTimes();

        int size = words.size();

        stringstream text;

        // Create JSON object
        for (int i = 0; i < size; i++) {
            json::JSON word;
            string w = model_->word_syms_->Find(words[i]);
            word["word"] = w;
            word["start"] = samples_round_start_ / sample_frequency_ + (frame_offset_ + times[i].first) * 0.03;
            word["end"] = samples_round_start_ / sample_frequency_ + (frame_offset_ + times[i].second) * 0.03;
            word["conf"] = conf[i];

            if (w.compare("<unk>") != 0)
                uttConfidence += conf[i];
            
            metadata_["words"].append(word);

            if (i) {
                text << " ";
            }
            text << model_->word_syms_->Find(words[i]);
        }
        uttConfidence /= size;
        
        if (metadata_["text"].ToString() == ""){
            metadata_["text"] = text.str();
        } else {
            stringstream text_;
            text_ << metadata_["text"].ToString() << " " << text.str();
            metadata_["text"] = text_.str();
        }
    }
    catch (std::bad_alloc& ba)
    {
        KALDI_WARN << ba.what() << " (No metadata is generated!)";
        metadata_.Make(json::JSON::Class::Null);
    }

}

const char* KaldiRecognizer::GetResult()
{
    if (decoder_->NumFramesDecoded() == 0) {
        return StoreReturn("{\"text\": \"\"}");
    }

    kaldi::CompactLattice clat;
    decoder_->GetLattice(true, &clat);

    if (model_->std_lm_fst_) {
        Lattice lat1;

        ConvertLattice(clat, &lat1);
        fst::ScaleLattice(fst::GraphLatticeScale(-1.0), &lat1);
        fst::ArcSort(&lat1, fst::OLabelCompare<kaldi::LatticeArc>());
        kaldi::Lattice composed_lat;
        fst::Compose(lat1, *lm_fst_, &composed_lat);
        fst::Invert(&composed_lat);
        kaldi::CompactLattice determinized_lat;
        DeterminizeLattice(composed_lat, &determinized_lat);
        fst::ScaleLattice(fst::GraphLatticeScale(-1), &determinized_lat);
        fst::ArcSort(&determinized_lat, fst::OLabelCompare<kaldi::CompactLatticeArc>());

        kaldi::ConstArpaLmDeterministicFst const_arpa_fst(model_->const_arpa_);
        kaldi::CompactLattice composed_clat;
        kaldi::ComposeCompactLatticeDeterministic(determinized_lat, &const_arpa_fst, &composed_clat);
        kaldi::Lattice composed_lat1;
        ConvertLattice(composed_clat, &composed_lat1);
        fst::Invert(&composed_lat1);
        DeterminizeLattice(composed_lat1, &clat);
    }

    if (clat.NumStates() == 0) {
        KALDI_WARN << "Empty lattice.";
        return StoreReturn("{\"text\": \"\"}");
    }

    kaldi::CompactLattice best_path_clat;
    kaldi::CompactLatticeShortestPath(clat, &best_path_clat);
   
    kaldi::Lattice best_path_lat;
    ConvertLattice(best_path_clat, &best_path_lat);

    kaldi::LatticeWeight weight;
    std::vector<int32> alignment;
    std::vector<int32> words;
    fst::GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);

    int size = words.size();

    stringstream text;

    for (int i = 0; i < size; i++) {
        if (i) {
            text << " ";
        }
        text << model_->word_syms_->Find(words[i]);
    }

    ComputeTimestamp(clat);

    return StoreReturn("{\"text\": \""+text.str()+"\"}");
}

const char* KaldiRecognizer::PartialResult()
{
    if (state_ != RECOGNIZER_RUNNING) {
        return StoreReturn("{\"partial\": \"\"}");
    }

    if (decoder_->NumFramesDecoded() == 0) {
        return StoreReturn("{\"partial\": \"\"}");
    }

    kaldi::Lattice lat;
    decoder_->GetBestPath(false, &lat);
    vector<kaldi::int32> alignment, words;
    LatticeWeight weight;
    GetLinearSymbolSequence(lat, &alignment, &words, &weight);

    ostringstream text;
    for (size_t i = 0; i < words.size(); i++) {
        if (i) {
            text << " ";
        }
        text << model_->word_syms_->Find(words[i]);
    }

    return StoreReturn("{\"partial\": \""+text.str()+"\"}");
}

const char* KaldiRecognizer::Result()
{
    if (state_ != RECOGNIZER_RUNNING) {
        return StoreReturn("{\"text\": \"\"}");
    }
    decoder_->FinalizeDecoding();
    state_ = RECOGNIZER_ENDPOINT;
    return GetResult();
}

const char* KaldiRecognizer::FinalResult()
{
    if (state_ != RECOGNIZER_RUNNING) {
        return StoreReturn("{\"text\": \"\"}");
    }

    feature_pipeline_->InputFinished();
    UpdateSilenceWeights();
    decoder_->AdvanceDecoding();
    decoder_->FinalizeDecoding();
    state_ = RECOGNIZER_FINALIZED;
    GetResult();

    return last_result_.c_str();
}

const char* KaldiRecognizer::GetMetadata()
{
    if (metadata_.IsNull()) {
        return last_result_.c_str();
    }
    StoreReturn(metadata_.dump());
    return last_result_.c_str();
}

// Store result in recognizer and return as const string
const char *KaldiRecognizer::StoreReturn(const string &res)
{
    last_result_ = res;
    return last_result_.c_str();
}
