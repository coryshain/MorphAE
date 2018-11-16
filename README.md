# Encoder-Decoder Morphology Learner

An encoder-decoder model of morphology acquisition.
A comprehension module attempts to recognize morphological and lexical features from inflected forms,
while a production module attempts to generate inflected froms lexical and morphological features.
The two modules are linked through a filter that is driven to mask all morphological features except those needed for successful inflection.

The model trains on tuples of <lexeme_id, inflected_form, morph_feats>.
Any of these elements can be seeded with noise in order to model referential uncertainty or comprehension difficulty during acquisition.

The model acquires a repurposeable form of morphological knowledge.
Its latent structures can be used to

- Generate inflections given a lexeme and morphological features
- Reconstruct inflected forms
- Classify inflected forms' morphological properties
- Infer the lexeme identity of inflected forms
- Generate citation forms from strings or lexeme IDs
- Reinflect arbitrary forms into a given paradigm cell


