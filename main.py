import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

# Carrega o dataset
df = pd.read_csv('docometria_da_pena.csv')

# Modelo de PLN (Classificação de Tipo de Crime)
def train_pln_model(df):
    # Cria um dicionário de palavras-chave para cada tipo de crime
    crime_keywords = {}
    for crime in df['Tipo_Crime'].unique():
        subset = df[df['Tipo_Crime'] == crime]
        all_words = ' '.join(subset['Observações']).split()
        common_words = [word for word, count in Counter(all_words).most_common(10)]
        crime_keywords[crime] = set(common_words)
    return crime_keywords

def predict_crime_type(observacao, crime_keywords):
    words = observacao.split()
    scores = {crime: sum(word in keywords for word in words) for crime, keywords in crime_keywords.items()}
    return max(scores, key=scores.get)

crime_keywords = train_pln_model(df)

# Modelo de Regressão (Previsão de Pena Aplicada)
def train_regression_model(df):
    # Calcula a média de pena por combinação de características
    averages = df.groupby(['Gênero_Réu', 'Tipo_Crime', 'Regime_Inicial', 'Reincidência'])['Pena_Aplicada_Years'].mean()
    return averages

def predict_sentence_length(genero, tipo_crime, regime_inicial, reincidencia, averages):
    key = (genero, tipo_crime, regime_inicial, reincidencia)
    if key in averages:
        return averages[key]
    else:
        return df['Pena_Aplicada_Years'].mean()  # Retorna a média geral se a combinação não for encontrada

averages = train_regression_model(df)

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
            prediction = predict_crime_type(observacao, crime_keywords)
            st.success(f"Tipo de Crime Previsto: {prediction}")

elif task == "Prever Pena Aplicada (Regressão)":
    st.subheader("Previsão de Pena Aplicada")
    idade = st.number_input("Idade do Réu:", min_value=18, max_value=100, value=25)
    genero = st.selectbox("Gênero do Réu:", ["Masculino", "Feminino"])
    tipo_crime = st.selectbox("Tipo de Crime:", ["Roubo", "Homicídio", "Tráfico de Drogas", "Furto", "Estupro"])
    regime_inicial = st.selectbox("Regime Inicial:", ["Fechado", "Semiaberto", "Aberto"])
    reincidencia = st.selectbox("Reincidência:", ["Sim", "Não"])
    
    if st.button("Prever"):
        prediction = predict_sentence_length(genero, tipo_crime, regime_inicial, reincidencia, averages)
        st.success(f"Pena Prevista: {prediction:.2f} anos")