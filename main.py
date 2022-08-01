import os
import numpy as np
import pandas as pd
from Ngram import Ngram
from model import softmax_regression

if __name__ == '__main__':
    train_df = pd.read_csv('./data/train.tsv', sep='\t')
    x_data, y_data = train_df["Phrase"].values, train_df["Sentiment"].values

    just_test = 1

    if just_test == 1:
        x_data = x_data[:20000]
        y_data = y_data[:20000]

    my_gram = Ngram(2)
    x_gram = my_gram.fit_transform(x_data)
    print(x_gram)
    my_softmax = softmax_regression()
    my_softmax.fit(sim=x_gram, res=y_data, learning_rate=0.01, batch_size=1, epochs=10)

    # train_df = pd.read_csv('./data/test.tsv', sep='\t')
    # x_data, y_data = train_df["PhraseId"].values, train_df["Phrase"].values
    # y_gram = my_gram.transform(y_data)
    # print(y_gram)
    # lis = []
    # for s in y_gram:
    #     lis.append(my_softmax.predict(s))
    # dataframe = pd.DataFrame({'PhraseId': x_data, 'Sentiment': lis})
    # dataframe.to_csv("./data/ans.csv", index=False, sep=',')

    train_df = pd.read_csv('./data/train.tsv', sep='\t')
    x_data, y_data = train_df["Phrase"].values, train_df["Sentiment"].values
    x_data = x_data[20000:40000]
    y_data = y_data[20000:40000]
    x_gram = my_gram.transform(x_data)
    cnt = 0
    for i, s in enumerate(x_gram):
        if my_softmax.predict(s) == y_data[i]:
            cnt += 1
    print(cnt/20000, '%')
