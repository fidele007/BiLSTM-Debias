#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>

#include <random>

using namespace std;
using namespace cnn;


//float pdrop = 0.5;
unsigned LAYERS = 1;
unsigned INPUT_DIM = 128;
unsigned HIDDEN_DIM = 128;
unsigned TAG_HIDDEN_DIM = 32;
unsigned TAG_DIM = 32;
unsigned TAG_SIZE = 0;
unsigned VOCAB_SIZE = 0;

bool eval = false;
cnn::Dict d;
cnn::Dict td;
int kNONE;
int kSOS;
int kEOS;

// default epochs
unsigned MAX_EPOCHS=10;

// use the universal tagset
const string TAG_SET[] = {"VERB", "NOUN", "PRON","ADJ", "ADV", "ADP", "CONJ", "DET", "NUM", "PRT", "X", "."};

template <class Builder>
struct RNNJointModel {
        LookupParameters* p_w;
        Parameters* p_l2th;
        Parameters* p_r2th;
        Parameters* p_thbias;

        Parameters* p_th2t;
        Parameters* p_tbias;
        Builder l2rbuilder;
        Builder r2lbuilder;

        // noise layer
        Parameters* p_nl;

        explicit RNNJointModel(Model& model) :
                l2rbuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
                r2lbuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {
                p_w = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM});
                p_l2th = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
                p_r2th = model.add_parameters({TAG_HIDDEN_DIM, HIDDEN_DIM});
                p_thbias = model.add_parameters({TAG_HIDDEN_DIM});

                p_th2t = model.add_parameters({TAG_SIZE, TAG_HIDDEN_DIM});
                p_tbias = model.add_parameters({TAG_SIZE});

                // noise layer
                p_nl = model.add_parameters({TAG_SIZE, TAG_SIZE});
        }

        Expression BuildTaggingGraphWithNoise(const vector<int>& sent, const vector<int>& tags, ComputationGraph& cg, double* cor = 0, unsigned* ntagged = 0, unsigned isNoNoise=0) {
                const unsigned slen = sent.size();
                l2rbuilder.new_graph(cg);
                l2rbuilder.start_new_sequence();
                r2lbuilder.new_graph(cg);
                r2lbuilder.start_new_sequence();
                Expression i_l2th = parameter(cg, p_l2th);
                Expression i_r2th = parameter(cg, p_r2th);
                Expression i_thbias = parameter(cg, p_thbias);
                Expression i_th2t = parameter(cg, p_th2t);
                Expression i_tbias = parameter(cg, p_tbias);
                vector<Expression> errs;
                vector<Expression> i_words(slen);
                vector<Expression> fwds(slen);
                vector<Expression> revs(slen);

                // read sequence from left to right
                l2rbuilder.add_input(lookup(cg, p_w, kSOS));
                for (unsigned t = 0; t < slen; ++t) {
                        i_words[t] = lookup(cg, p_w, sent[t]);
                        if (!eval) { i_words[t] = noise(i_words[t], 0.1); }
                        fwds[t] = l2rbuilder.add_input(i_words[t]);
                }

                // read sequence from right to left
                r2lbuilder.add_input(lookup(cg, p_w, kEOS));
                for (unsigned t = 0; t < slen; ++t)
                        revs[slen - t - 1] = r2lbuilder.add_input(i_words[slen - t - 1]);

                for (unsigned t = 0; t < slen; ++t) {
                        if (tags[t] != kNONE) {
                                if (ntagged) (*ntagged)++;
                                Expression i_th = tanh(affine_transform({i_thbias, i_l2th, fwds[t], i_r2th, revs[t]}));
                                Expression i_t = affine_transform({i_tbias, i_th2t, i_th});
                                if (cor) {
                                        vector<float> dist = as_vector(cg.incremental_forward());
                                        double best = -9e99;
                                        int besti = -1;
                                        for (int i = 0; i < dist.size(); ++i) {
                                                if (dist[i] > best) { best = dist[i]; besti = i; }
                                        }
                                        if (tags[t] == besti) (*cor)++;
                                }
                                // different objectives
                                if(isNoNoise==1) {
                                        //Expression i_a = const_parameter(cg, p_nl); //const but no use
                                        Expression i_err = pickneglogsoftmax(i_t, tags[t]);
                                        errs.push_back(i_err);

                                } else {
                                        Expression i_a = parameter(cg, p_nl);
                                        Expression i_nl = i_a * i_t;
                                        Expression i_nl_err = pickneglogsoftmax(i_nl, tags[t]);
                                        errs.push_back(i_nl_err);
                                }

                        }
                }
                return sum(errs);
        }

        // predict the tags of an inpute sentence
        vector<string> PredictSequentTags(const vector<int>& sent, const vector<int>& tags, ComputationGraph& cg, double* cor = 0, unsigned* ntagged = 0) {
                const unsigned slen = sent.size();
                l2rbuilder.new_graph(cg); // reset RNN builder for new graph
                l2rbuilder.start_new_sequence();
                r2lbuilder.new_graph(cg); // reset RNN builder for new graph
                r2lbuilder.start_new_sequence();
                Expression i_l2th = parameter(cg, p_l2th);
                Expression i_r2th = parameter(cg, p_r2th);
                Expression i_thbias = parameter(cg, p_thbias);
                Expression i_th2t = parameter(cg, p_th2t);
                Expression i_tbias = parameter(cg, p_tbias);
                vector<Expression> errs;
                vector<Expression> i_words(slen);
                vector<Expression> fwds(slen);
                vector<Expression> revs(slen);

                vector<string> preds;

                // read sequence from left to right
                l2rbuilder.add_input(lookup(cg, p_w, kSOS));
                for (unsigned t = 0; t < slen; ++t) {
                        i_words[t] = lookup(cg, p_w, sent[t]);
                        fwds[t] = l2rbuilder.add_input(i_words[t]);
                }
                // read sequence from right to left
                r2lbuilder.add_input(lookup(cg, p_w, kEOS));
                for (unsigned t = 0; t < slen; ++t)
                        revs[slen - t - 1] = r2lbuilder.add_input(i_words[slen - t - 1]);

                for (unsigned t = 0; t < slen; ++t) {

                        if (ntagged) (*ntagged)++;
                        Expression i_th = tanh(affine_transform({i_thbias, i_l2th, fwds[t], i_r2th, revs[t]}));

                        Expression i_t = affine_transform({i_tbias, i_th2t, i_th});
                        if (cor) {
                                vector<float> dist = as_vector(cg.incremental_forward());
                                double best = -9e99;
                                int besti = -1;
                                for (int i = 0; i < dist.size(); ++i) {
                                        if (dist[i] > best) { best = dist[i]; besti = i; }
                                }
                                if (tags[t] == besti) (*cor)++;
                        }
                        double best = 9e+99;
                        string ptag;
                        for(const string& tag: TAG_SET) {
                                Expression i_err_t = pickneglogsoftmax(i_t, td.Convert(tag));
                                double error = as_scalar(i_err_t.value());
                                if(error < best) {
                                        best = error;
                                        ptag= tag;
                                }

                        }
                        preds.push_back(ptag);
                }
                return preds;
        }
};

int main(int argc, char** argv) {
        cnn::Initialize(argc, argv);
        if (argc != 6) {
          cerr << "Usage: " << argv[0] << " gold_data projected_data model_params file_to_tag tagged_file_output\n";
          return 1;
        }
        kNONE = td.Convert("*");
        // use universal tagset
        for(const string& tag: TAG_SET) {
                td.Convert(tag);
        }
        td.Freeze(); // no new tag types allowed
        TAG_SIZE = td.size();

        kSOS = d.Convert("<s>");
        kEOS = d.Convert("</s>");
        vector<pair<vector<int>, vector<int> > > training, dev, test;
        vector<unsigned> data_type;

        string line;
        int tlc = 0;
        int ttoks = 0;
        cerr << "Reading supervision data from " << argv[1] << "...\n";
        {
                ifstream in(argv[1]);
                assert(in);
                while(getline(in, line)) {
                        ++tlc;
                        //read both the words and tags
                        vector<int> x, y;
                        ReadSentencePair(line, &x, &d, &y, &td);
                        assert(x.size()==y.size());
                        if (x.size()==0) {cerr << line << endl; abort(); }
                        training.push_back(make_pair(x,y));
                        // no noise
                        data_type.push_back(1);
                        ttoks += x.size();

                }
                cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
        }

        tlc = 0;
        ttoks = 0;
        cerr << "Reading projection data from " << argv[2] << "...\n";
        {
                ifstream in(argv[2]);
                assert(in);
                while(getline(in, line)) {
                        ++tlc;
                        //read both the words and tags
                        vector<int> x, y;
                        ReadSentencePair(line, &x, &d, &y, &td);
                        assert(x.size()==y.size());
                        if (x.size()==0) {cerr << line << endl; abort(); }
                        training.push_back(make_pair(x,y));
                        // exist noise
                        data_type.push_back(0);
                        ttoks += x.size();

                }
                cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
        }

        d.Freeze(); // no new word types allowed
        d.SetUnk("<UNK>");
        VOCAB_SIZE = d.size();
        assert(training.size()==data_type.size());

        Model model;
        bool use_momentum = true;
        Trainer* sgd = nullptr;
        float lambda=1e-3;
        if (use_momentum)
                sgd = new MomentumSGDTrainer(&model, lambda);
        else
                sgd = new SimpleSGDTrainer(&model);


        RNNJointModel<LSTMBuilder> lm(model);

        // Import previous model parameters
        string fname = argv[3];
        cerr << "Reading params from " << fname << "..." << endl;
        ifstream in(fname);
        boost::archive::text_iarchive ia(in);
        ia >> model;
        cerr << "Got the model params." << endl;

        // Predict a new text
        vector<pair<vector<int>, vector<int>>> unknown;

        // // For testing only
        // string snt1 = "Haiti : Manavotra ny velona , Mitady ny tsy hita ||| X X X X X X X X X X";
        // string snt2 = "Christopher Frecynet mbola velona . ||| X X X X X";

        // vector<int> x, y;
        // ReadSentencePair(snt1, &x, &d, &y, &td);
        // assert(x.size()==y.size());
        // unknown.push_back(make_pair(x,y));

        // vector<int> a, b;
        // ReadSentencePair(snt2, &a, &d, &b, &td);
        // assert(a.size()==b.size());
        // unknown.push_back(make_pair(a,b));

        vector<vector<string>> text;
        vector<vector<string>> tags;
        string eachLine;
        {
                ifstream in(argv[4]);
                assert(in);
                while(getline(in, eachLine)) {
                        // Store sentence
                        vector<string> snt;
                        boost::split(snt, eachLine, boost::is_any_of("\t "));
                        text.push_back(snt);
                        string formattedSnt = eachLine + " ||| ";
                        for (unsigned it = 0; it < snt.size(); it++) {
                                formattedSnt += "X ";
                        }
                        boost::trim_left(formattedSnt);
                        boost::trim_right(formattedSnt);

                        cout << "Processing sentence: " << formattedSnt << endl;
                        vector<int> x, y;
                        ReadSentencePair(formattedSnt, &x, &d, &y, &td);
                        assert(x.size()==y.size());
                        if (x.size()==0) {cerr << formattedSnt << endl; abort(); }
                        unknown.push_back(make_pair(x,y));
                }
        }

        cout << "Start tagging..." << endl;
        for (auto& sent : unknown) {
                ComputationGraph cg;
                vector<string> preds=lm.PredictSequentTags(sent.first, sent.second, cg);
                // for (unsigned i=0; i < sent.first.size(); i++) {
                //         cout << preds[i] << " ";
                // }
                // cout << endl;

                // Store tags
                tags.push_back(preds);
        }

        // Write tagged text to |result|
        assert(text.size()==tags.size());
        string result = "";
        for (unsigned i = 0; i < text.size(); i++) {
                vector<string> sent = text[i];
                vector<string> tag = tags[i];
                for (unsigned j = 0; j < sent.size(); j++) {
                        result += sent[j] + "|" + tag[j] + " ";
                }
                result += "\n";
                // cout << result;
        }

        // Write result to file
        string outputFileName = argv[5];
        ofstream out(outputFileName);
        out << result;
        out.close();

        cout << "Finished tagging. Tagged file exported to " << outputFileName << endl;

        delete sgd;
        return 0;
}
