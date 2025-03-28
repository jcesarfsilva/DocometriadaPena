import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Carrega o dataset
df = pd.read_csv('docometria_da_pena.csv')

# Modelo de PLN
vectorizer = TfidfVectorizer(max_features=1000)
X_pln = vectorizer.fit_transform(df['Observações'])
y_pln = df['Tipo_Crime']
model_pln = MultinomialNB()
model_pln.fit(X_pln, y_pln)

# Modelo de Regressão
features = ['Idade_Réu', 'Gênero_Réu', 'Tipo_Crime', 'Regime_Inicial', 'Reincidência']
X_reg = df[features]
y_reg = df['Pena_Aplicada_Years']
categorical_features = ['Gênero_Réu', 'Tipo_Crime', 'Regime_Inicial', 'Reincidência']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)
model_regressao = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
model_regressao.fit(X_reg, y_reg)

# Interface Streamlit
st.title("Análise de Docometria da Pena")
st.write("Selecione uma tarefa abaixo:")

# Opções
task = st.selectbox("Escolha uma tarefa:", ["Classificar Tipo de Crime (PLN)", "Prever Pena Aplicada (Regressão)"])

if task == "Classificar Tipo de Crime (PLN)":
    st.subheader("Classificação de Tipo de Crime")
    observacao = st.text_area("Insira a observação do caso:")
    if st.button("Classificar"):
        if observacao.strip() == "":
            st.warning("Por favor, insira uma observação.")
        else:
            input_vector = vectorizer.transform([observacao])
            prediction = model_pln.predict(input_vector)[0]
            st.success(f"Tipo de Crime Previsto: {prediction}")

elif task == "Prever Pena Aplicada (Regressão)":
    st.subheader("Previsão de Pena Aplicada")
    idade = st.number_input("Idade do Réu:", min_value=18, max_value=100, value=25)
    genero = st.selectbox("Gênero do Réu:", ["Masculino", "Feminino"])
    tipo_crime = st.selectbox("Tipo de Crime:", ["Roubo", "Homicídio", "Tráfico de Drogas", "Furto", "Estupro"])
    regime_inicial = st.selectbox("Regime Inicial:", ["Fechado", "Semiaberto", "Aberto"])
    reincidencia = st.selectbox("Reincidência:", ["Sim", "Não"])
    
    if st.button("Prever"):
        input_data = pd.DataFrame([[idade, genero, tipo_crime, regime_inicial, reincidencia]], columns=features)
        prediction = model_regressao.predict(input_data)[0]
        st.success(f"Pena Prevista: {prediction:.2f} anos")




# pandas==2.2.3
# numpy==1.26.4
# scipy==1.15.2
# scikit-learn==1.6.1
# streamlit==1.38.0
# Faker==37.1.0