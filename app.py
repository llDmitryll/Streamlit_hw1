import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error as MSE

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model, scaler, feature_names = pickle.load(f)
    return model, scaler, feature_names

# Загружаем модель
try:
    MODEL, SCALER, FEATURE_NAMES = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()


# --- Основной интерфейс ---
st.title("🎯 Предсказание стоимости автомобилей")

# Загрузка CSV файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if uploaded_file is None:
    st.info("👈 Загрузите CSV файл для начала работы")
    st.stop()

# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file)
df = df.drop('Unnamed: 0', axis=1)

try:
    features = df[FEATURE_NAMES]
    y_pred = MODEL.predict(SCALER.transform(features))
except Exception as e:
    st.error(f"❌ Ошибка при обработке данных: {e}")
    st.stop()


st.subheader("📊 Pairplot")
fig = sns.pairplot(df)
st.pyplot(fig)

st.subheader("📊 Correlation Heatmap")
corr_df = df.corr(numeric_only=True)
fig, ax = plt.subplots()
sns.heatmap(corr_df)
st.pyplot(fig)

st.subheader("Результаты предсказаний")
df['prediction'] = y_pred
a = pd.concat([df['selling_price'], pd.DataFrame(y_pred.astype(int), columns=['prediction'])], axis=1)
st.dataframe(a)

st.dataframe(pd.DataFrame({'MSE': [MSE(df['selling_price'], y_pred)], 'R2 Score': [r2_score(df['selling_price'], y_pred)]}))

# --- Метрики ---
st.subheader("Веса модели")

w = pd.concat([pd.Series(features.columns), pd.Series(MODEL.coef_)], axis=1)
st.dataframe(w)
