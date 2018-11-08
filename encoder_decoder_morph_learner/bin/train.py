import sys
import os
import shutil
import argparse

sys.setrecursionlimit(2000)

from encoder_decoder_morph_learner.config import Config
from encoder_decoder_morph_learner.kwargs import ENCODER_DECODER_MORPH_LEARNER_INITIALIZATION_KWARGS
from encoder_decoder_morph_learner.model import EncoderDecoderMorphLearner

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains an encoder-decoder morphology learner model from a config file.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    args = argparser.parse_args()

    p = Config(args.config)

    if not p['use_gpu_if_available']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    sys.stderr.write('Initializing encoder-decoder...\n\n')

    if args.restart and os.path.exists(p.outdir + '/tensorboard'):
        shutil.rmtree(p.outdir + '/tensorboard')

    kwargs = {}

    for kwarg in ENCODER_DECODER_MORPH_LEARNER_INITIALIZATION_KWARGS:
        kwargs[kwarg.key] = p[kwarg.key]

    morph_feature_map = {}

    edml_model = EncoderDecoderMorphLearner(
        morph_feature_map,
        **kwargs
    )
