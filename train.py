import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
import pandas as pd
from pandas import DataFrame
import seaborn as sb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import f_classif

def edit_df(df):
    df.dropna(inplace=True)
    for index, row in df.iterrows():
        # В исходных данных присутствуют записи с типом данных datetime в столбце Наследственность. Удаляем их
        if type(row["Наследственность"]).__name__ == "datetime":
            df.drop(index, inplace=True)
    df["Наследственность"] = df["Наследственность"].astype(float)
    return df

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

#Получаем исходные данные из Excel файла
diabet_df: DataFrame = pd.read_excel(r'C:\Users\User\PycharmProjects\ml_pr_1\diabetes.xlsx')
#Удалили некорректные данные
diabet_df = edit_df(diabet_df)

#Независимые переменные
X = diabet_df[["Беременность", "Глюкоза", "АД", "Толщина КС", "Инсулин", "ИМТ", "Наследственность", "Возраст"]]
#Зависимая переменная
y = diabet_df["Диагноз"]

#Разделяем набор данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = LogisticRegression(lr=0.01) #Классиффикатор
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

acc = accuracy(y_pred, y_test)*100
print(f'Accuracy = {acc} %')

# Корреляция
correlations = diabet_df.corr()
sb.heatmap(correlations, linewidths=0.5, annot=True,
           cmap='viridis', annot_kws={'size': 9})
plt.show()

# Мультиколлинеарность
vif = [variance_inflation_factor(X_train.values, i) for i in
       range(X_train.shape[1])]
print(pd.DataFrame({'vif': vif[0:]}, index=X_train.columns).T)
cols_names = ["Беременность", "Глюкоза", "АД", "Толщина КС", "Инсулин", "ИМТ", "Наследственность", "Возраст",
              "Диагноз"]
cfs_score = f_classif(X, y)
print(pd.DataFrame(list(zip(cols_names, cfs_score[0], cfs_score[1])), columns=['ftr', 'score', 'pval']))

sb.pairplot(diabet_df)
plt.show()

# Изменим входной набор данных, т.к. Артериальное давление и Толщина КС продемонстрировали маленький результат
new_diabet_df = diabet_df.loc[:, ~diabet_df.columns.isin(['АД', 'Толщина КС'])]

# Проведем переобучение
X = new_diabet_df[new_diabet_df.columns[:-1]]
y = new_diabet_df[new_diabet_df.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = LogisticRegression(lr=0.01)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

acc = accuracy(y_pred, y_test)*100
print(f'New Accuracy = {acc} %')

#В результате более тщательного отбора признаков значение Accuracy выросло (63.39869281045751% против 66.01307189542483%)
