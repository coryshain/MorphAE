import sys
import os
import pickle
import argparse

from encoder_decoder_morph_learner.config import Config
from encoder_decoder_morph_learner.util import load_edml

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Interactively generate reconstructions and morphological classifications for user-provided forms. 
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    args = argparser.parse_args()

    p = Config(args.config)

    if not p['use_gpu_if_available']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    edml_model = load_edml(p.outdir)

    prompt = True
    reinflect = False

    sys.stderr.write('''
    Welcome to the interactive mode for the encoder-decoder morphology learner!
    To use, type an inflected word form into the prompt below.
    The learner will attempt to report the morphological features associated with that form, and to use those features to reconstruct it.
    To exit, type "Q".\n\n''')

    while prompt:
        if reinflect:
            s = input('Type a space-delimited list of reinflection features (optional) >>> ')
            if s.strip() == '':
                reinflect = False
            else:
                reinflections.append(s.strip().split())
        else:
            s = input('Type an inflected word >>> ')
            input_form = s.strip()
            reinflect = True
            reinflections = []

        if s.strip() == 'Q':
            prompt = False
        elif not reinflect:
            forms, morph_feats, lexeme = edml_model.process_string(input_form, reinflections=reinflections)
            sys.stderr.write('  Results:')
            sys.stderr.write('\n    Lexeme: %s\n' % lexeme)
            sys.stderr.write('    Predicted morph features: %s\n' % morph_feats[0])
            sys.stderr.write('    Reconstruction: %s\n' % forms[0])
            sys.stderr.write('    Citation form: %s\n' % forms[1])
            if len(forms) > 2:
                sys.stderr.write('    Reinflections:\n')
                for i in range(2, len(forms)):
                    sys.stderr.write('      %s | %s\n' %(morph_feats[i], forms[i]))
            sys.stderr.write('\n')


    sys.stderr.write('\nOk, goodbye!\n')

    edml_model.finalize()
