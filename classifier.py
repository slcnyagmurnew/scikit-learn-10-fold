import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import argparse
import warnings
import csv
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')

"""
maps for categorical data to convert number and give them to classifiers
"""
column_map = {'₺400-999': 0, '₺1.000-3.999': 1, '₺4.000-6.999': 2, '₺7.000-11.999': 3, '₺12.000+': 4,
              'Bilişim-Yazılım': 0, 'Eğitim': 1, 'Sağlık': 2, 'Makine-Sanayi': 3, 'Elektrik-Elektronik': 4,
              'Sosyal Hizmetler': 5, 'İşletme-Yönetim': 6, 'Finans': 7, 'Denizcilik': 8,
              'Hayır': 0, 'Evet': 1, 'Erkek': 0, 'Kadın': 1, '18-23': 0, '24-29': 1, '30-39': 2, '40-49': 3, '50+': 4,
              'İş yerinde': 0, 'Hibrit': 1, 'Uzaktan': 2
              }


def rename_columns():
    """
    Read raw data and rename raw data columns like 'Cinsiyetiniz nedir?' to 'gender'
    :return: dataframe: with renamed columns
    """
    df = pd.read_csv('data/yz_anket_V2.csv')
    df.columns = ['time', 'gender', 'age', 'income', 'work', 'love_job', 'marital_status', 'have_pet', 'satisfy_income',
                  'work_type', 'satisfy_city', 'have_car', 'smoking_or_alcohol', 'horoscope']
    return df


def mapping(df):
    """
    Map column values to integer or float values
    :param df: current dataframe
    :return: current dataframe after mapping
    """
    df = df.stack().map(column_map).unstack()
    return df


def plot_confusion_matrix(testY, y_pred):
    """
    Draw confusion matrix for all algorithms' last accuracy scores
    :param testY: real data for an input
    :param y_pred: predicted data for an input
    :return:
    """
    cm = confusion_matrix(testY, y_pred, normalize='true')
    print(cm)
    cm_df = pd.DataFrame(cm, columns=[0, 1, 2, 3, 4, 5, 6], index=[0, 1, 2, 3, 4, 5, 6])
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    heatmap(cm_df, cmap='Blues', annot=True)
    plt.show()


def evaluate_metric(testY, y_pred):
    """
    Get evaluation metrics(precision, recall etc.) of different types of classification methods.
    :param testY: float; y_test value to evaluate accuracy score.
    :param y_pred: float; prediction value of each classifier using x_test data.
    :return:
    """
    report = classification_report(testY, y_pred)
    print("Classification Report:", )
    print(report)
    score = accuracy_score(testY, y_pred)
    print("Accuracy:", score)
    return score


def get_model(k):
    """
    Get selected method and return it with parameters from scikit-learn library
    :param k: selected method's abbreviation
    :return: model: classifier model
    """
    if k == 'd':
        return DecisionTreeClassifier()
    elif k == 'r':
        return RandomForestClassifier(n_estimators=10)
    elif k == 'l':
        return LogisticRegression(solver='liblinear')
    elif k == 's':
        return SVC(kernel='linear')
    else:
        return MultinomialNB()


def csv_to_text(file, out):
    """
    Convert csv file to text file
    :param file: input csv file
    :param out: output text file
    :return:
    """
    with open(out, "w") as o:
        with open(file, "r") as i:
            [o.write(" ".join(row) + '\n') for row in csv.reader(i)]
        o.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", required=True, help='selected method for classification')

    args = vars(parser.parse_args())
    method = args['method']  # get method from terminal

    # data = rename_columns()  # rename columns only first time
    # data.to_csv('data/yz_anket.csv', index=None)  # save csv with renamed column names
    data = pd.read_csv('data/yz_anket.csv')
    data = mapping(data)

    X = data[['gender', 'age', 'work_type']]
    y = data['work']
    # X = data[['age', 'work', 'income']]
    # y = data['work_type']
    # X = data[['income', 'work']]
    # y = data['satisfy_income']

    k = 10
    classifier = get_model(method)
    kf = KFold(n_splits=k, random_state=None)  # create cross validation constructor
    acc_score = []  # get 10 accuracy score for all pass

    fold = 0
    for train_index, test_index in kf.split(X):
        print(f'-------------------------KFold {fold}-------------------------')
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)
        pred_values = classifier.predict(X_test)

        acc = evaluate_metric(y_test, pred_values)
        acc_score.append(acc)

        fold += 1
        if fold == 10:
            plot_confusion_matrix(y_test, pred_values)

    # csv_to_text('data/yz_anket.csv', 'data/yz_anket.txt')  # save data as text file
