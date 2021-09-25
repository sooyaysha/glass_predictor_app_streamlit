# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Droping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)

    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

@st.cache()
def prediction(model, ri, na, mg, al, si, k, ca, ba, fe):
    glass_type = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
    glass_type = glass_type[0]
    if glass_type == 1:
        return "building windows float processed".upper()
    elif glass_type == 2:
        return "building windows non float processed".upper()
    elif glass_type == 3:
        return "vehicle windows float processed".upper()
    elif glass_type == 4:
        return "vehicle windows non float processed".upper()
    elif glass_type == 5:
        return "containers".upper()
    elif glass_type == 6:
        return "tableware".upper()
    else:
        return "headlamps".upper()

# Add title on the main page and in the sidebar.
st.title("Glass Type Predictor")
st.sidebar.title("Exploratory Data Analysis")

# Using if statement, display raw data on the click of the checkbox.
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Full Dataset")
    st.dataframe(glass_df)

# Sidebar for scatter plot.
st.sidebar.subheader("Scatter Plot")

# Choosing x-axis values for scatter plots.
features_list = st.sidebar.multiselect("Select the x-axis values:", 
                                            ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

# Remove deprecation warning.
st.set_option('deprecation.showPyplotGlobalUse', False)

# Create scatter plots.
for feature in features_list:
    st.subheader(f"Scatter plot between {feature} and GlassType")
    plt.figure(figsize = (12, 6))
    sns.scatterplot(x = feature, y = 'GlassType', data = glass_df)
    st.pyplot()

st.sidebar.subheader('Visualization Selector')
plot_types = st.sidebar.multiselect('Select the Charts/Plots: ', ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))

if 'Histogram' in plot_types:
  st.subheader("Histogram")
  hist_columns = st.sidebar.selectbox("Select the column to create its histogram:", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize = (12, 6))
  plt.title(f'Histogram for {hist_columns}')
  plt.hist(glass_df[hist_columns], bins = 'sturges', edgecolor = 'black')
  st.pyplot() 

if 'Box Plot' in plot_types:
  st.subheader("Box Plot")
  box_columns = st.sidebar.selectbox("Select the column to create its boxplot:", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize = (12, 2))
  plt.title(f'Boxplot for {box_columns}')
  sns.boxplot(glass_df[box_columns])
  st.pyplot() 

if 'Count Plot' in plot_types:
  st.subheader('Count Plot')
  sns.countplot(x = 'GlassType', data = glass_df)
  st.pyplot()
# Create pie chart using the 'matplotlib.pyplot' module and the 'st.pyplot()' function.   
if 'Pie Chart' in plot_types:
  st.subheader('Pie Chart')
  pie_data = glass_df['GlassType'].value_counts()
  plt.figure(figsize = (5,5))
  plt.pie(pie_data, labels = pie_data.index, autopct = '%1.2f%%', startangle = 30, explode = np.linspace(0.06, 0.16, 6))
  st.pyplot()
# Display correlation heatmap using the 'seaborn' module and the 'st.pyplot()' function.
if 'Correlation Heatmap' in plot_types:
  st.subheader('Correlation Heatmap')
  plt.figure(figsize = (10,6))
  ax = sns.heatmap(glass_df.corr(), annot = True)
  bottom, top = ax.get_ylim()
  ax.set_ylim(bottom + 0.5, top - 0.5)
  st.pyplot()
# Display pair plots using the the 'seaborn' module and the 'st.pyplot()' function. 
if 'Pair Plot' in plot_types:
  st.subheader('Pair Plots')
  plt.figure(figsize = (15,15))
  sns.pairplot(glass_df)
  st.pyplot()

st.sidebar.subheader("Select your values:")
ri = st.sidebar.slider("Input Ri", float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider("Input Na", float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider("Input Mg", float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider("Input Al", float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider("Input Si", float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k = st.sidebar.slider("Input K", float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider("Input Ca", float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider("Input Ba", float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider("Input Fe", float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))

st.sidebar.subheader('Choose Classifier')
classifier = st.sidebar.selectbox('Classifier: ', ('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))

if classifier == 'Support Vector Machine':
  st.sidebar.subheader('Model Hyper-Parameters')
  c_value = st.sidebar.number_input('C (Error-Rate)', 1,100, step = 1)
  kernel_input = st.sidebar.radio('Kernel', ('linear', 'rbf', 'poly'))
  gamma_input = st.sidebar.number_input('Gamma', 1,10, step = 1)
  if st.sidebar.button('Classify'):
    st.subheader('Support Vector Machine')
    svc_model = SVC(C = c_value, kernel = kernel_input, gamma = gamma_input)
    svc_model.fit(X_train, y_train)
    y_pred = svc_model.predict(X_test)
    accuracy = svc_model.score(X_test, y_test)
    glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
    st.write(f'Type of Glass Predicted: {glass_type}')
    st.write('Accuracy: ', accuracy.round(2))
    plot_confusion_matrix(svc_model, X_test, y_test)
    st.pyplot()

if classifier == 'Random Forest Classifier':
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
    max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1, 100, step = 1)

    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement. 
    if st.sidebar.button('Classify'):
        st.subheader("Random Forest Classifier")
        rf_clf = RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
        rf_clf.fit(X_train,y_train)
        accuracy = rf_clf.score(X_test, y_test)
        glass_type = prediction(rf_clf, ri, na, mg, al, si, k, ca, ba, fe)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(rf_clf, X_test, y_test)
        st.pyplot()