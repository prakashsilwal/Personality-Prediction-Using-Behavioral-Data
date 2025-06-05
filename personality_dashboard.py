import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("personality_dataset.csv")
    return df

df = load_data()

# 🔁 Apply preprocessing globally
# Clean missing values
df['Time_spent_Alone'].fillna(df['Time_spent_Alone'].mean(), inplace=True)
df['Social_event_attendance'].fillna(df['Social_event_attendance'].mean(), inplace=True)
df['Going_outside'].fillna(df['Going_outside'].mean(), inplace=True)
df['Friends_circle_size'].fillna(df['Friends_circle_size'].mean(), inplace=True)
df['Post_frequency'].fillna(df['Post_frequency'].mean(), inplace=True)
df['Stage_fear'].fillna(df['Stage_fear'].mode()[0], inplace=True)
df['Drained_after_socializing'].fillna(df['Drained_after_socializing'].mode()[0], inplace=True)

# Encode categorical columns
df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
df['Personality'] = df['Personality'].map({'Extrovert': 1, 'Introvert': 0})

# ------------------ Sidebar ------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Preprocessing", "EDA", "Model Comparison"])

# ------------------ Section 1: Preprocessing ------------------
if section == "Preprocessing":
    st.title("🧼 Data Preprocessing")
    st.subheader("Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.dataframe(df.head())

    st.subheader("Missing Values After Cleaning")
    st.write(df.isnull().sum())

    st.subheader("Encoded Data Sample")
    st.dataframe(df.head())

# ------------------ Section 2: EDA ------------------
elif section == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Personality Type Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Personality', ax=ax1)
    st.pyplot(fig1)

    st.subheader("Stage Fear by Personality")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x='Stage_fear', hue='Personality', ax=ax2)
    st.pyplot(fig2)

    st.subheader("Drained After Socializing by Personality")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=df, x='Drained_after_socializing', hue='Personality', ax=ax3)
    st.pyplot(fig3)

    st.subheader("Correlation Heatmap")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)

# ------------------ Section 3: Model Comparison ------------------
elif section == "Model Comparison":
    st.title("🤖 Model Accuracy Comparison")

    # Prepare data
    X = df.drop(columns=['Personality', 'Friends_circle_size'])
    y = df['Personality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale for relevant models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": DecisionTreeClassifier(random_state=42),  # Replace with XGBoost if available
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(kernel='linear', random_state=42)
    }

    accuracies = []
    for name, model in models.items():
        if name in ["Logistic Regression", "KNN", "SVM"]:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append((name, acc))

    # Plot accuracy comparison
    acc_df = pd.DataFrame(accuracies, columns=["Model", "Accuracy"])
    st.dataframe(acc_df.sort_values(by="Accuracy", ascending=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=acc_df, x="Model", y="Accuracy", palette="crest", ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)

    best_model = acc_df.loc[acc_df['Accuracy'].idxmax()]
    st.success(f"Best Performing Model: {best_model['Model']} with Accuracy: {best_model['Accuracy']*100:.2f}%")
