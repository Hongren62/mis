import mysql.connector
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 使用spaCy 英文停用詞
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# 連接DB
db = mysql.connector.connect(
    host="localhost",
    port=3306,
    database="mydb",
    user="root",
    password="brian0602"
)

mycursor = db.cursor()

# SELECT資料
mycursor.execute("SELECT `Publication Number`, `Assignee/Applicant`, `Title (English)`, `Abstract (English)`, `Claims (English)`, `Publication Date`, `Application Date`, `Priority Date`, `CPC - Current`, `CPC - Current - DWPI`, `Assignee/Applicant First`, `Assignee - Standardized`, `Assignee - Original`, `Assignee - Original w/address`, `Assignee - Original - Country/Region`, `Assignee - Current US`, `Optimized Assignee`, `Assignee - DWPI`, `Front Page Drawing`, `Front Page Image`, `Inventor`, `US Class`, `DWPI Class`, `DWPI Family Members`, `Priority Number`, `Priority Number - DWPI` FROM `f01l_patent-mis`")
data = mycursor.fetchall()

# 把資料轉成DataFrame
columns = ["Publication Number", "Assignee/Applicant", "Title (English)", "Abstract (English)", "Claims (English)", "Publication Date", "Application Date", "Priority Date", "CPC - Current", "CPC - Current - DWPI", "Assignee/Applicant First", "Assignee - Standardized", "Assignee - Original", "Assignee - Original w/address", "Assignee - Original - Country/Region", "Assignee - Current US", "Optimized Assignee", "Assignee - DWPI", "Front Page Drawing", "Front Page Image", "Inventor", "US Class", "DWPI Class", "DWPI Family Members", "Priority Number", "Priority Number - DWPI"]
df = pd.DataFrame(data, columns=columns)

mycursor.close()

# 把專利欄位用spaCy停用詞做預處理
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha and token.text.lower() not in stop_words]
    return ' '.join(tokens)
df['Processed Abstract'] = df['Abstract (English)'].apply(preprocess_text)

print(df['DWPI Class'].value_counts())

# 選擇樣本數最多的前10類
top_classes = df['DWPI Class'].value_counts().index[:10]
df_top_classes = df[df['DWPI Class'].isin(top_classes)]

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df_top_classes['Processed Abstract'])

# 把TF-IDF结果加到DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
df_top_classes = pd.concat([df_top_classes.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

# PCA
pca = PCA(n_components=2)
tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())

# 把PCA結果加到DataFrame
pca_df = pd.DataFrame(tfidf_pca, columns=['PCA Component 1', 'PCA Component 2'])
df_top_classes = pd.concat([df_top_classes, pca_df], axis=1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix.toarray(), df_top_classes['DWPI Class'], test_size=0.2, random_state=42)

# Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)

# SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# 混淆矩阵和分類結果
def evaluate_model(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for {model_name}:\n", cm)
    print(f"Classification Report for {model_name}:\n", classification_report(y_test, y_pred))
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=top_classes, yticklabels=top_classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

evaluate_model(y_test, y_pred_rf, "Random Forest")

evaluate_model(y_test, y_pred_svm, "SVM")
