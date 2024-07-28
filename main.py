import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import imaplib
import email

# Veri setinin yüklenmesi ve temizlenmesi
df = pd.read_csv("spam.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Eğitim ve test verilerinin ayrılması
x_train, x_test, y_train, y_test = train_test_split(df["Message"], df["Category"], test_size=0.2, random_state=42)

# Metin verisinin sayısal verilere dönüştürülmesi
cv = CountVectorizer()
x_train_vectorized = cv.fit_transform(x_train)
x_test_vectorized = cv.transform(x_test)

# Modelin eğitilmesi
lg = LogisticRegression()
lg.fit(x_train_vectorized, y_train)

# Modelin değerlendirilmesi
y_pred = lg.predict(x_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 score: {f1:.4f}')

# E-posta hesabına bağlanma
user = 'mail'
password = '$$'

imap_url = 'imap.gmail.com'
my_mail = imaplib.IMAP4_SSL(imap_url)
my_mail.login(user, password)
my_mail.select('Inbox')
typ, data = my_mail.search(None, 'ALL')
mail_id_list = data[0].split()

# E-postaların alınması ve işlenmesi
num_emails_to_process = min(20, len(mail_id_list))
for num in mail_id_list[-num_emails_to_process:]:
    typ, data = my_mail.fetch(num, '(RFC822)')
    msg = email.message_from_bytes(data[0][1])
    
    sender = msg['From']
    subject = msg['Subject']
    body = None
    
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True).decode()
                break
    else:
        body = msg.get_payload(decode=True).decode()
    
    if body is None:
        print("Skipping email with no plain text body")
        continue

    # E-postaların sınıflandırılması
    input_data = [body]
    input_data_features = cv.transform(input_data)
    prediction = lg.predict(input_data_features)
    
    print("From:", sender)
    print("Subject:", subject)
    print("Spam" if prediction[0] == 'spam' else "Not Spam")
    print("---------------------------------------")
