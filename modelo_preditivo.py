import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Carregar os dados
df = pd.read_csv('dataSet.csv', sep=';')

# 2. Pré-processamento
# Separar variáveis preditoras (X) e variável alvo (y)
X = df.drop(columns=['SITUDESC'])  
y = df['SITUDESC']

# Codificar a variável alvo 'SITUDESC' com LabelEncoder (para transformar as classes em números)
le = LabelEncoder()
y = le.fit_transform(y)

# Identificar colunas categóricas automaticamente
categorical_columns = X.select_dtypes(include=['object']).columns

# Transformar variáveis categóricas em numéricas (One-Hot Encoding)
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# 3. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Treinar o modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Avaliar o modelo
y_pred = model.predict(X_test)

# Imprimir a matriz de confusão
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Imprimir o relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
