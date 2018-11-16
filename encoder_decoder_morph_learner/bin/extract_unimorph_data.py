import sys
import os
import pickle
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Training, cross-validation, and testing data from a unimorph language repository.
        ''')
    argparser.add_argument('path', help='Path to unimorph directory.')
    argparser.add_argument('language', help='Three-letter language code.')
    argparser.add_argument('-o', '--outdir', type=str, default=None, help='Path to file in which to save extracted list')
    args = argparser.parse_args()

    l = args.language
    if args.outdir is None:
        outdir = './unimorph_data/%s' %l
    else:
        outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if l == 'fin':
        filenames = ['fin.1', 'fin.2']
    else:
        filenames = [l]

    train_data = []
    cv_data = []
    test_data = []

    for name in filenames:
        with open(args.path + '/' + l + '/' + name, 'r', encoding='utf8') as unifile:
            for i, line in enumerate(unifile):
                if line.strip() != '':
                    lemma, form, feats = line.strip().split('\t')
                    feats = feats.split(';')
                    if i % 4 == 0:
                        data = test_data
                    elif i % 4 == 1:
                        data = cv_data
                    else:
                        data = train_data

                    data.append((lemma, form, feats))

    with open(outdir + '/' + 'train_data.obj', 'wb') as outfile:
        pickle.dump(train_data, outfile)
    with open(outdir + '/' + 'cv_data.obj', 'wb') as outfile:
        pickle.dump(cv_data, outfile)
    with open(outdir + '/' + 'test_data.obj', 'wb') as outfile:
        pickle.dump(test_data, outfile)