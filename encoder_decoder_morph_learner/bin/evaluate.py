import sys
import os
import pickle
import argparse

from encoder_decoder_morph_learner.config import Config
from encoder_decoder_morph_learner.util import load_edml

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Evaluate EDML model on a given dataset.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    argparser.add_argument('-n', '--n_eval', type=int, default=None, help='Number of data points to evaluate on. If left unspecified, evaluates on the entire set.')
    args = argparser.parse_args()

    p = Config(args.config)

    if not p['use_gpu_if_available']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    edml_model = load_edml(p.outdir)

    prompt = True

    if args.partition.lower() == 'train':
        with open(p.train_data, 'rb') as f:
            data = pickle.load(f)
    elif args.partition.lower() in ['dev', 'cv']:
        with open(p.cv_data, 'rb') as f:
            data = pickle.load(f)
    elif args.partition.lower() == 'test':
        with open(p.test_data, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError('Partition type "%s" not recognized.' % args.partition)

    _, _, reconstruction_failures = edml_model.evaluate_reconstructions(data, return_errors=True)

    outfile = edml_model.outdir + '/failures_' + args.partition + '.txt'
    with open(outfile, 'w') as f:
        f.write(reconstruction_failures)

    edml_model.evaluate_reinflections(data, n_eval=args.n_eval)
