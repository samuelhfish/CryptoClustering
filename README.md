# CryptoClustering
Demo using unsupervised machine learning, K-means, clustering and PCA (Principal Component Analysis) to predict if cryptocurrencies are affected by 24-hour or 7-day price changes.

The goal of this exercise is to:

- prepare data using StandardScaler from scikit learn
- find the best value for k uisng the scaled dataframe
- cluster the Cryptocurrencies with K-means using the scaled data
- optimize clusters using Principal Component Analysis (PCA)
- find best value for k with PCA data
- cluster again with K-means and PCA data
- compare results

All code exists in the Crypto_Clustering Jupyter Notebook inside the CryptoClustering folder. Source data (crypto_market_data.csv) exists inside the Resources folder.

```
After importing depedencies we have our DataFrame, df_market_data
```
![Screenshot 2023-12-18 at 8 00 56 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/99d7ee89-f822-4048-8e37-8c1d3653c6d5)
```python
# Generate summary statistics
df_market_data.describe()
```
![Screenshot 2023-12-18 at 8 01 55 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/8cb2c4ae-cd21-4a23-8333-081b84aa4e30)
```python
# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)
```
![Screenshot 2023-12-18 at 8 02 49 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/4564a4d6-fe42-4e2d-a934-0a8fa4aa61a7)

```python
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
market_data_scaled = StandardScaler().fit_transform(df_market_data[["price_change_percentage_24h",
                                                            "price_change_percentage_7d",
                                                            "price_change_percentage_14d",
                                                            "price_change_percentage_30d",
                                                            "price_change_percentage_60d",
                                                            "price_change_percentage_200d",
                                                            "price_change_percentage_1y"]])

# Create a DataFrame with the scaled data
df_market_data_transformed = pd.DataFrame(market_data_scaled, columns=["price_change_percentage_24h",
                                                            "price_change_percentage_7d",
                                                            "price_change_percentage_14d",
                                                            "price_change_percentage_30d",
                                                            "price_change_percentage_60d",
                                                            "price_change_percentage_200d",
                                                            "price_change_percentage_1y"])

# Copy the crypto names from the original data
df_market_data_transformed["coin_id"] = df_market_data.index

# Set the coinid column as index
df_market_data_transformed = df_market_data_transformed.set_index("coin_id")

# Display sample data
df_market_data_transformed.head(10)
```
![Screenshot 2023-12-18 at 8 03 42 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/a3816521-597a-4e57-bf9d-1b572aa32d30)
```python
# Find the Best Value for k Using the Original Data.
# Create a list with the number of k-values from 1 to 11
k = list(range(1, 11))

# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_transformed`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    k_model = KMeans(n_clusters=i, random_state=0)
    k_model.fit(df_market_data_transformed)
    inertia.append(k_model.inertia_)

# Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)


# Create a DataFrame with the data to plot the Elbow curve
df_elbow.head()
```
![Screenshot 2023-12-18 at 8 04 39 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/49d4be44-b2fc-4819-9931-650ee902e2cb)

```python
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_elbow_plot = df_elbow.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
)

df_elbow_plot
```
![Screenshot 2023-12-18 at 8 05 36 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/e3382fe0-3c17-4632-acd8-b759bbdc5962)

Our line plot tells us that 4 is the best value for 'k'.

```python
# Now we cluster using K-Means Method
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4,)

# Fit the K-Means model using the scaled data
model.fit(df_market_data_transformed)
```
```
KMeans(n_clusters=4)
```
```python
# Predict the clusters to group the cryptocurrencies using the scaled data
k_4 = model.predict(df_market_data_transformed)

# Print the resulting array of cluster values.
k_4
```
```
array([2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2,
       0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 0, 0, 3, 0, 0, 0, 0],
      dtype=int32)
```
```python
# Create a copy of the DataFrame
df_market_data_predictions = df_market_data_transformed.copy()

# Add a new column to the DataFrame with the predicted clusters
df_market_data_predictions['cluster_prediction'] = k_4

# Display sample data
df_market_data_predictions.head(10)
```
![Screenshot 2023-12-18 at 8 08 42 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/551efe28-52ea-4f66-a48f-97eb469eccb4)
```python
# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
df_market_data_predictions_plot = df_market_data_predictions.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by="cluster_prediction",
    hover_cols = ["coin_id"]
)

df_market_data_predictions_plot
```
![Screenshot 2023-12-18 at 8 09 47 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/386f6189-fd1f-4399-8395-70ba19ad7c3d)

```python
# Optimize Clusters with Principal Component Analysis.
# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)

# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
market_data_pca = pca.fit_transform(df_market_data_transformed)

# View the first five rows of the array. 
market_data_pca[:5]
```
```
array([[-0.60066733,  0.84276006,  0.46159457],
       [-0.45826071,  0.45846566,  0.95287678],
       [-0.43306981, -0.16812638, -0.64175193],
       [-0.47183495, -0.22266008, -0.47905316],
       [-1.15779997,  2.04120919,  1.85971527]])
```
```python
# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_
```
```
array([0.3719856 , 0.34700813, 0.17603793])
```
Total explained variance of the three princiapl comonents is .895
```python
# Create a new DataFrame with the PCA data.
df_pca = pd.DataFrame(market_data_pca, columns=["PCA1", "PCA2", "PCA3"])

# Copy the crypto names from the original data
df_pca["coin_id"] = df_market_data.index

# Set the coinid column as index
df_pca = df_pca.set_index("coin_id")

# View the first five rows of the DataFrame. 
df_pca.head(5)
```
![Screenshot 2023-12-18 at 10 25 42 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/57c01d01-c6a3-4ccb-b371-3468de20e6a6)
```python
# Find the Best Value for k Using the PCA Data
# Create a list with the number of k-values from 1 to 11
k = list(range(1, 11))

# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
for i in k:
    # Inside the loop:
    # 1. Create a KMeans model using the loop counter for the n_clusters
    k_model = KMeans(n_clusters=i, random_state=1)
    # 2. Fit the model to the data using `df_market_data_pca`
    k_model.fit(df_pca)
    # 3. Append the model.inertia_ to the inertia list
    inertia.append(k_model.inertia_)

# Create a dictionary with the data to plot the Elbow curve
elbow_data_pca = {"k": k, "inertia": inertia}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow_pca = pd.DataFrame(elbow_data_pca)

df_elbow_pca.head()
```
![Screenshot 2023-12-18 at 10 26 35 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/564c9584-17bd-4dca-b57d-adfaeca445e3)
```python
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_elbow_pca_plot = df_elbow_pca.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
)

df_elbow_pca_plot
```
![Screenshot 2023-12-18 at 10 27 56 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/1b8b8b51-bec1-4a51-a92f-df03a69fd88b)

It looks like 4 is still the best value for 'k' when using the PCA data which is the same as with our original data.

Let's now repeat the clustering with K-means using the PCA data.

```python
# Initialize the K-Means model using the best value for k
model_pca = KMeans(n_clusters=4)

# Fit the K-Means model using the PCA data
model_pca.fit(df_pca)
```
```
KMeans(n_clusters=4)
```
```python
# Predict the clusters to group the cryptocurrencies using the PCA data
k_4_pca = model_pca.predict(df_pca)

# Print the resulting array of cluster values.
k_4_pca
```
```
array([1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 3, 0, 0, 0, 0],
      dtype=int32)
```
```python
# Create a copy of the DataFrame with the PCA data
df_pca_predictions = df_pca.copy()

# Add a new column to the DataFrame with the predicted clusters
df_pca_predictions["cluster_pca"] = k_4_pca

# Display sample data
df_pca_predictions.head()
```
![Screenshot 2023-12-18 at 10 30 21 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/3069441b-f326-4723-b075-024228aee69c)
```python
# Create a scatter plot using hvPlot by setting 
# `x="PC1"` and `y="PC2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
df_pca_predictions_plot = df_pca_predictions.hvplot.scatter(
    x="PCA1",
    y="PCA2",
    by="cluster_pca",
    hover_cols = ["coin_id"]
)

df_pca_predictions_plot
```
![Screenshot 2023-12-18 at 10 31 32 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/132920cb-aa33-4a52-bd26-1c56fd808b24)

Now we visualize and compare the results by contrasting the outcome with and without using the optimization techniques.
```python
# Composite plot to contrast the Elbow curves
df_elbow_plot * df_elbow_pca_plot
```
![Screenshot 2023-12-18 at 10 35 29 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/b9e6e11e-b682-4099-a8b4-710369d21a73)
```python
# Composite plot to contrast the clusters
df_market_data_predictions_plot + df_pca_predictions_plot
```
![Screenshot 2023-12-18 at 10 36 41 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/f44ed45d-db00-4dfc-b498-0d904b567a19)
![Screenshot 2023-12-18 at 10 36 52 PM](https://github.com/samuelhfish/CryptoClustering/assets/125224990/d08d753d-d2c6-4870-a9a5-d872499b1254)

#### Analysis:
After visually analyzing the cluster analysis results, we see that using fewer features allows us to more easily identify outlier. Fewer features also lowered the inertia on our elbow curve.