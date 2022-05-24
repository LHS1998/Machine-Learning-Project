import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np

train_df = pd.read_csv('./data/train.csv', sep='\t')
test_df = pd.read_csv('./data/test.csv', sep='\t')


def word_tokenizer(string):  # 一般文本列分词
    words = nltk.tokenize.word_tokenize(string)  # 先切词
    # stemmer = nltk.stem.SnowballStemmer('english') # 使用nltk词干化工具
    # words = [stemmer.stem(w) for w in words] # 对每个词词干化
    lemma = nltk.wordnet.WordNetLemmatizer()
    words = [lemma.lemmatize(w).lower() for w in words]
    stopwords = nltk.corpus.stopwords.words('english')  # 使用nltk的停用词
    words = [w for w in words if w not in stopwords]  # 去除停用词
    return words


docs = train_df["reviewText"].tolist()  # 取出所有文本字符串
docy = test_df["reviewText"].tolist()
vectorizer = TfidfVectorizer(tokenizer=word_tokenizer)
train_x = vectorizer.fit_transform(docs)  # 文本向量
train_x = hstack((train_x, train_df[["reviewerID", "asin", "overall"]])).tocsr()  # 加入其他特征
test_x = vectorizer.transform(docy)  # 测试特征使用训练集上生成的语料库，这样类似真实环境，而且训练集没有出现的句子对测试集也没有帮助
test_x = hstack((test_x, test_df[["reviewerID", "asin", "overall"]])).tocsr()  # 加入其他特征

np.savez("train_x.npz", data=train_x.data, indices=train_x.indices,
         indptr=train_x.indptr, shape=train_x.shape)
np.savez("test_x.npz", data=test_x.data, indices=test_x.indices,
         indptr=test_x.indptr, shape=test_x.shape)
