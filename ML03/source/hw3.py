import jsonpickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer  # 提取文本特征向量的类
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB  # 三种朴素贝叶斯算法，差别在于估计p(x|y)的方式
from sklearn.model_selection import train_test_split  # 划分数据集函数
import Mail

with open("splited_for_script.json", mode='rt') as f:
    mail_data = jsonpickle.decode(f.read())

texts = []
labels = []
for mail in mail_data:
    text = " ".join(mail.nouns) + " ".join(mail.verbs) + " ".join(mail.other) + " ".join(mail.title_split) + mail.ip_
    if mail.from_add.find("@") != -1:
        text += mail.from_add.split("@")[1]
    # text = " ".join(mail.nouns) + " ".join(mail.verbs) + " ".join(mail.other)
    # text = " ".join(mail.nouns) + " ".join(mail.verbs) + " ".join(mail.adjvs) + " ".join(mail.other)
    label = 1 if mail.label == 'spam' else 0
    texts.append(text)
    labels.append(label)

vectorizer = TfidfVectorizer()
vectorizer.fit(texts)
vectors = vectorizer.transform(texts)

x_train, x_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2)


def acc(y, y_hat):
    true_ = p_true = p_real = p_hat = 0.
    for y_i, y_i_hat in zip(y, y_hat):
        if y_i == 1:  # 计数 TP+FP
            p_real += 1
        if y_i_hat == 1:  # 计数 TP+FN
            p_hat += 1
        if y_i == y_i_hat:
            true_ += 1
            if y_i == 1:  # 计数 TP
                p_true += 1
    accuracy = true_ / len(y)
    precision = p_true / p_real
    recall = p_true / p_hat
    return accuracy, precision, recall


def acc2(y, y_hat):
    accuracy0, precision0, recall0 = acc(y, y_hat)
    p_true = p_real = p_hat = 0.
    for y_i, y_i_hat in zip(y, y_hat):
        if y_i == 0:  # 计数 TP+FP
            p_real += 1
        if y_i_hat == 0:  # 计数 TP+FN
            p_hat += 1
        if y_i == y_i_hat:
            if y_i == 0:  # 计数 TP
                p_true += 1
    precision = p_true / p_real
    recall = p_true / p_hat
    return accuracy0, precision0, recall0, precision, recall


clf = MultinomialNB()
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
accu1 = acc2(y_test, y_predict)

clf2 = BernoulliNB()
clf2.fit(x_train, y_train)
y_predict2 = clf2.predict(x_test)
accu2 = acc2(y_test, y_predict2)

clf3 = ComplementNB()
clf3.fit(x_train, y_train)
y_predict3 = clf3.predict(x_test)
accu3 = acc2(y_test, y_predict3)

print("|   Model_Name  |  Accuracy  |  Precision |   Recall   | ~Precision |   ~Recall  |")
print("---------------------------------------------------------------------------------")
s_format = "|%10.4f  |%10.4f  |%10.4f  |%10.4f  |%10.4f  |"
print("|%14s" % "MultinomialNB", s_format % accu1)
print("|%14s" % "BernoulliNB", s_format % accu2)
print("|%14s" % "ComplementNB", s_format % accu3)
