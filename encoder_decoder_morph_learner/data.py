import numpy as np

from .util import get_random_permutation


def get_data_generator(data, char_to_ix, morph_to_ix, lex_to_ix, max_seq_len=25, randomize=True):
    n_data = len(data)
    i = 0
    if randomize:
        ix, ix_inv = get_random_permutation(n_data)
    else:
        ix = np.arange(n_data)

    preprocessing_function = get_data_preprocessing_function(
        char_to_ix,
        morph_to_ix,
        lex_to_ix,
        max_seq_len=max_seq_len
    )

    while True:
        if i >= n_data:
            i = 0
            if randomize:
                ix, ix_inv = get_random_permutation(n_data)

        lexeme, form, form_mask, morph_feat = preprocessing_function(data[ix[i]])
        i += 1

        yield lexeme, form, form_mask, morph_feat


def get_data_preprocessing_function(char_to_ix, morph_to_ix, lex_to_ix, max_seq_len=25):
    n_char = len(char_to_ix)
    n_morph = len(morph_to_ix)
    n_lex = len(lex_to_ix)

    def preprocessing_function(data_point):
        lexeme = data_point[0]
        lexeme_out = np.zeros((n_lex,))
        if lexeme in lex_to_ix:
            lex_ix = lex_to_ix[lexeme]
        else:
            lex_ix = -1
        lexeme_out[lex_ix] = 1

        form_str = data_point[1]
        offset = max_seq_len - len(form_str)
        form_out = np.zeros((max_seq_len, n_char))
        form_out[:,-1] = 1
        form_mask_out = np.zeros((max_seq_len,))
        for k, c in enumerate(form_str):
            form_out[k, -1] = 1
            form_out[k, char_to_ix[c]] = 1
        if offset > 0:
            form_mask_out[:-offset] = 1
        else:
            form_mask_out[:] = 1

        morph_feat_str = data_point[2]
        morph_feat_out = np.zeros((n_morph,))
        for m in morph_feat_str:
            morph_feat_out[morph_to_ix[m]] = 1

        return lexeme_out, form_out, form_mask_out, morph_feat_out

    return preprocessing_function


def reconstruct_characters(char_probs, char_set):
    out = []
    indices = np.argmax(char_probs, axis=-1)
    for w in indices:
        cur = ''
        for i in w:
            if char_set[i] is not None:
                cur += char_set[i]
        out.append(cur)

    return out


def reconstruct_morph_feats(morph_feat_probs, morph_set):
    out = []
    morph_feats_discrete = morph_feat_probs > 0.5
    for w in morph_feats_discrete:
        m_feats = []
        for j, p in enumerate(w):
            if p:
                m_feats.append(morph_set[j])
        out.append(';'.join(m_feats))

    return out


def stringify_data(form_gold, form_pred, morph_gold, morph_pred, char_set=None, morph_set=None):
    if not isinstance(form_gold, list):
        assert char_set, 'If gold forms are given as one-hot, char_set must be provided.'
        form_gold =  reconstruct_characters(form_gold, char_set)

    if not isinstance(form_pred, list):
        assert char_set, 'If predicted forms are given as one-hot, char_set must be provided.'
        form_pred = reconstruct_characters(form_pred, char_set)

    if not isinstance(morph_gold, list):
        assert char_set, 'If gold morphs are given as multi-hot, morph_set must be provided.'
        morph_gold = reconstruct_morph_feats(morph_gold, morph_set)

    if not isinstance(morph_pred, list):
        assert char_set, 'If predicted morphs are given as multi-hot, morph_set must be provided.'
        morph_pred = reconstruct_morph_feats(morph_pred, morph_set)

    max_len_reconst = 0
    max_len_morph = 0
    for x in form_gold:
        max_len_reconst = max(len(x), max_len_reconst)
    for x in form_pred:
        max_len_reconst = max(len(x), max_len_reconst)
    for x in morph_gold:
        max_len_morph = max(len(x), max_len_morph)
    for x in morph_pred:
        max_len_morph = max(len(x), max_len_morph)

    out_str = ''
    for i in range(len(form_gold)):
        out_str += '    GOLD: %s | %s\n' %(morph_gold[i] + ' ' * (max_len_morph - len(morph_gold[i])), form_gold[i] + ' ' * (max_len_reconst - len(form_gold[i])))
        out_str += '    PRED: %s | %s\n\n' %(morph_pred[i] + ' ' * (max_len_morph - len(morph_pred[i])), form_pred[i] + ' ' * (max_len_reconst - len(form_pred[i])))

    return out_str
