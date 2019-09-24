import os
import os.path
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble
import spacy
import sys
from sklearn.feature_extraction.text import CountVectorizer
from anchor_zh import anchor_text
import jieba
import crash_on_ipy
from sklearn.externals import joblib


def load_pingan3(fpath, idx2lbl=None):
    data = []
    labels = []
    with open(fpath, 'r') as f:
        for line in f:
            lbl, txt = line.strip().split('\t')
            txt = ' '.join(jieba.cut(txt.strip()))
            data.append(txt)
            labels.append(lbl)
    if idx2lbl is None:
        idx2lbl = list(set(labels))
    lbl2idx = {lbl: idx for idx, lbl in enumerate(idx2lbl)}
    for i in range(len(labels)):
        labels[i] = lbl2idx[labels[i]]

    return data, labels, idx2lbl, lbl2idx


ftrain = os.path.join('notebooks/pingan3', 'train_data.txt')
fvalid = os.path.join('notebooks/pingan3', 'validation_data.txt')

train, train_lbls, idx2lbl, lbl2idx = load_pingan3(ftrain)
valid, valid_lbls, _, _ = load_pingan3(fvalid, idx2lbl)

vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)
train_vectors = vectorizer.transform(train)
valid_vectors = vectorizer.transform(valid)

c = sklearn.ensemble.RandomForestClassifier(n_estimators=10, n_jobs=3,
                                            random_state=0, class_weight='balanced')
c.fit(train_vectors, train_lbls)


def predict_lr(texts):
    return c.predict(vectorizer.transform(texts))


preds = c.predict(valid_vectors)
print('Valid f1', sklearn.metrics.f1_score(valid_lbls, preds, average='macro'))
preds = c.predict(train_vectors)
print('Train f1', sklearn.metrics.f1_score(train_lbls, preds, average='macro'))

nlp = spacy.load('zh')
explainer = anchor_text.AnchorText(nlp, idx2lbl, use_unk_distribution=False)
sidx = 3
np.random.seed(1)
text = valid[sidx]
text = ''.join(text.split())
pred = explainer.class_names[predict_lr([text])[0]]
alternative = explainer.class_names[1 - predict_lr([text])[0]]
print('Input: %s' % text)
print('Prediction: %s' % pred)
exp = explainer.explain_instance(text, predict_lr, threshold=0.95, use_proba=True, verbose=False)

print('Anchor: %s' % (' AND '.join(exp.names())))
print('Precision: %.2f' % exp.precision())
print()
print('Examples where anchor applies and model predicts %s:' % pred)
print()
print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
print()
print('Examples where anchor applies and model predicts others')
print()
print('\n'.join(['\t'.join((x[0], idx2lbl[predict_lr([x[0]])[0]]))
                 for x in exp.examples(only_different_prediction=True)]))
