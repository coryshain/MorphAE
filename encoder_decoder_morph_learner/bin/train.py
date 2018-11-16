import sys
import os
import pickle
import shutil
import argparse
import string

sys.setrecursionlimit(2000)

from encoder_decoder_morph_learner.config import Config
from encoder_decoder_morph_learner.kwargs import ENCODER_DECODER_MORPH_LEARNER_INITIALIZATION_KWARGS
from encoder_decoder_morph_learner.model import EncoderDecoderMorphLearner

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains an encoder-decoder morphology learner model from a config file.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    argparser.add_argument('-r', '--restart', action='store_true', help='Restart training even if model checkpoint exists (this will overwrite existing checkpoint)')
    args = argparser.parse_args()

    p = Config(args.config)

    if not p['use_gpu_if_available']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if args.restart and os.path.exists(p.outdir + '/tensorboard'):
        shutil.rmtree(p.outdir + '/tensorboard')

    kwargs = {}

    for kwarg in ENCODER_DECODER_MORPH_LEARNER_INITIALIZATION_KWARGS:
        kwargs[kwarg.key] = p[kwarg.key]

    morph_set = set()
    with open(p.unimorph_feature_file, 'r', encoding='utf8') as f:
        for l in f:
            if l.strip() != '':
                morph_set.add(l.strip())
    morph_set = sorted(list(morph_set))

    with open(p.train_data, 'rb') as f:
        train_data = pickle.load(f)
    with open(p.cv_data, 'rb') as f:
        cv_data = pickle.load(f)
    with open(p.test_data, 'rb') as f:
        test_data = pickle.load(f)

    lex_set = set()
    char_set = set()
    for x in train_data:
        lex_set.add(x[0])
        for c in x[1]:
            char_set.add(c)
    for x in cv_data:
        for c in x[1]:
            char_set.add(c)
    for x in test_data:
        for c in x[1]:
            char_set.add(c)
    lex_set = sorted(list(lex_set))
    char_set = sorted(list(char_set))

    sys.stderr.write('Initializing encoder-decoder...\n\n')

    edml_model = EncoderDecoderMorphLearner(
        morph_set,
        lex_set,
        char_set,
        **kwargs
    )

    edml_model.build()

    edml_model.fit(
        train_data,
        cv_data=cv_data,
        n_iter=p['n_iter']
    )
