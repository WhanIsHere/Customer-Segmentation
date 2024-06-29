import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set Streamlit page config to full width
st.set_page_config(layout="wide")

# Read the dataset from the CSV file
df = pd.read_csv('Dataset/Mall_Customers.csv') 
df = df.drop(['CustomerID'], axis=1)
X3 = df.iloc[:,1:]



# Streamlit app
st.title(':blue[Customer Segmentation] 🎯')
st.subheader(':blue[This project is to cluster the customers into different group based on their demographic in business.]', divider='rainbow')

st.write("**:blue[To show existing customers' information.]**") 
# Button to show the DataFrame
if st.button(':green[Show Dataset]'):
    st.write("**:blue[Dataset:]**")
    st.write(df)

wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(X3)
    wcss.append(kmeans.inertia_)

fig = plt.figure(figsize=(12,6))
plt.grid(False)
plt.plot(range(1,11),wcss, linewidth=2, color='red', marker='8')
plt.scatter(5, wcss[4], color='green', s=100)  
plt.xlabel("K Value")
plt.ylabel("WCSS")   
plt.title("Elbow Method for Optimal K")

st.subheader('', divider='rainbow')
text1= "Find the best number of cluster"
st.markdown(f"<h3 style='text-align: center;'>{text1}</h3>", unsafe_allow_html=True) 

# Displaying the plot in Streamlit
st.pyplot(fig)

text2 = 'Number of K-Cluster: 5'
st.markdown(f"<h3 style='text-align: center; color: green;'>{text2}</h3>", unsafe_allow_html=True)
  
st.subheader('', divider='rainbow')

# Train a KMeans model
kmeans = KMeans(n_clusters=5)
kmeans.fit(X3)
df['Cluster'] = kmeans.labels_    
    
# User Input Section
st.subheader("**:blue[Please input the new customer's information.]**")    
    
# User inputs
age = st.slider(':blue[Age]', min_value=0, max_value=100, value=25)
annual_income = st.slider(':blue[Annual Income (k$)]', min_value=0, max_value=150, value=50)
spending_score = st.slider(':blue[Spending Score]', min_value=0, max_value=100, value=50)

# Predict the cluster for the user input
user_data = np.array([[age, annual_income, spending_score]])
user_cluster = kmeans.predict(user_data)

st.subheader('', divider='rainbow')
st.title(':blue[Result] 💡')
# Centered text
centered_text = f'The customer belongs to cluster {user_cluster[0]}'
# Using HTML to center the text
st.markdown(f"<h3 style='text-align: center; color: blue;'>{centered_text}</h3>", unsafe_allow_html=True)


# Plotting the 3D scatter plot
fig2 = plt.figure(figsize=(30,20))
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.Cluster == 0], df['Annual Income (k$)'][df.Cluster == 0],df['Spending Score (1-100)'][df.Cluster == 0], c='blue', s=60)
ax.scatter(df.Age[df.Cluster == 1], df['Annual Income (k$)'][df.Cluster == 1],df['Spending Score (1-100)'][df.Cluster == 1], c='red', s=60)
ax.scatter(df.Age[df.Cluster == 2], df['Annual Income (k$)'][df.Cluster == 2],df['Spending Score (1-100)'][df.Cluster == 2], c='green', s=60)
ax.scatter(df.Age[df.Cluster == 3], df['Annual Income (k$)'][df.Cluster == 3],df['Spending Score (1-100)'][df.Cluster == 3], c='orange', s=60)
ax.scatter(df.Age[df.Cluster == 4], df['Annual Income (k$)'][df.Cluster == 4],df['Spending Score (1-100)'][df.Cluster == 4], c='purple', s=60)

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score')
ax.set_title('Customer Segmentation 3D Scatter Plot')

# Plotting the user input point
ax.scatter(age, annual_income, spending_score, color='cyan', s=100, label='User Input', edgecolor='k')

# Displaying the plot in Streamlit
st.pyplot(fig2)