import sys
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Extracts superset of morphological features from a directory of unimorph databases
        ''')
    argparser.add_argument('path', help='Path to unimorph directory.')
    argparser.add_argument('-o', '--outfile', type=str, default='./unimorph_feats.txt', help='Path to file in which to save extracted list')
    args = argparser.parse_args()

    langs = [
        'ara',
        'bul',
        'ces',
        'cym',
        'deu',
        'eng',
        'eus',
        'fas',
        'fra',
        'gle',
        'heb',
        'hun',
        'isl',
        'kat',
        'kmr',
        'lav',
        'mkd',
        'nld',
        'nob',
        'por',
        'ron',
        'slv',
        'spa',
        'swe',
        'ukr',
        'ben',
        'cat',
        'ckb',
        'dan',
        'dsb',
        'est',
        'fao',
        'fin',
        'gla',
        'hai',
        'hin',
        'hye',
        'ita',
        'klr',
        'lat',
        'lit',
        'nav',
        'nno',
        'pol',
        'que',
        'rus',
        'sme',
        'sqi',
        'tur',
        'urd'
    ]

    all_feats = set()

    for l in langs:
        sys.stderr.write('Extracting from language %s...\n' %l)
        if l == 'fin':
            filenames = ['fin.1', 'fin.2']
        else:
            filenames = [l]
        for name in filenames:
            with open(args.path + '/' + l + '/' + name, 'r', encoding='utf8') as unifile:
                for line in unifile:
                    if line.strip() != '':
                        _, _, feats = line.strip().split('\t')
                        feats = feats.split(';')
                        for f in feats:
                            all_feats.add(f.strip())

    all_feats = sorted(list(all_feats))

    with open(args.outfile, 'w') as outfile:
        for f in all_feats:
            outfile.write(f + '\n')