# Regression Analysis on Diamond Dataset

---

Author: [Mo Gutale](https://github.com/mgutale)

<img src="https://media.giphy.com/media/eHvd40K8OfCRRLB3t5/giphy.gif" width="950" height="280" />

---

## Brief Outline 

Businesses often nowadays require to predict prices of products based on a various product attributes and along with other methods in order to ensure that products are priced correctly. This notebook will aim to solve this business problem using the famous diamond dataset. By the end of this notebook, i am expecting to have a model that can be used by business to deploy in a real world scenario. 

## Dataset

This classic dataset contains the prices and other attributes of almost 54,000 diamonds. The source of this dataset can be found [here.](https://www.kaggle.com/datasets/shivam2503/diamonds) 

Features include:

- price in US dollars - Target
- carat weight of the diamond 
- cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- color diamond colour, from J (worst) to D (best)
- clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
- depth (43-79)
- x length in mm (0--10.74)
- y width in mm (0--58.9)
- z depth in mm (0--31.8)

The last three variables represent the dimension of the the particular observation(diamond).

---

## Table of Content

1. Data Load & Quick Look 
2. Explore, Clean & Transform 
3. Feature Selection and Preprocessing
4. Model Training 
5. HyperTunning the Model
6. Feature Importance
7. Prediction on Test & Confidence Interval
8. Save Model 
9. Prediction on Validation set
10. Conclusion 



---



## 1. Data Load & Quick Look


```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import time
warnings.filterwarnings('ignore')

# Read the file 
file_path = 'dataset/diamonds.csv'
df = pd.read_csv(os.path.join(os.getcwd(), file_path))

# Quick look at a random sample of 10
df.sample(10)
```





  <div id="df-252e1941-271d-49e0-8027-a0df545347e0">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53237</th>
      <td>53238</td>
      <td>0.72</td>
      <td>Ideal</td>
      <td>H</td>
      <td>VS1</td>
      <td>61.6</td>
      <td>59.0</td>
      <td>2642</td>
      <td>5.75</td>
      <td>5.78</td>
      <td>3.55</td>
    </tr>
    <tr>
      <th>13039</th>
      <td>13040</td>
      <td>0.36</td>
      <td>Ideal</td>
      <td>F</td>
      <td>SI1</td>
      <td>61.5</td>
      <td>56.0</td>
      <td>600</td>
      <td>4.59</td>
      <td>4.62</td>
      <td>2.83</td>
    </tr>
    <tr>
      <th>29728</th>
      <td>29729</td>
      <td>0.31</td>
      <td>Very Good</td>
      <td>G</td>
      <td>VVS2</td>
      <td>62.9</td>
      <td>58.0</td>
      <td>707</td>
      <td>4.28</td>
      <td>4.30</td>
      <td>2.70</td>
    </tr>
    <tr>
      <th>30120</th>
      <td>30121</td>
      <td>0.32</td>
      <td>Premium</td>
      <td>G</td>
      <td>VS2</td>
      <td>60.7</td>
      <td>58.0</td>
      <td>720</td>
      <td>4.42</td>
      <td>4.38</td>
      <td>2.67</td>
    </tr>
    <tr>
      <th>32482</th>
      <td>32483</td>
      <td>0.30</td>
      <td>Ideal</td>
      <td>G</td>
      <td>VS1</td>
      <td>61.8</td>
      <td>56.0</td>
      <td>795</td>
      <td>4.32</td>
      <td>4.35</td>
      <td>2.68</td>
    </tr>
    <tr>
      <th>5259</th>
      <td>5260</td>
      <td>0.80</td>
      <td>Ideal</td>
      <td>F</td>
      <td>VS1</td>
      <td>61.2</td>
      <td>56.0</td>
      <td>3793</td>
      <td>6.01</td>
      <td>5.98</td>
      <td>3.67</td>
    </tr>
    <tr>
      <th>11700</th>
      <td>11701</td>
      <td>0.34</td>
      <td>Premium</td>
      <td>E</td>
      <td>SI1</td>
      <td>60.7</td>
      <td>60.0</td>
      <td>596</td>
      <td>4.48</td>
      <td>4.51</td>
      <td>2.73</td>
    </tr>
    <tr>
      <th>29037</th>
      <td>29038</td>
      <td>0.40</td>
      <td>Good</td>
      <td>E</td>
      <td>SI1</td>
      <td>63.8</td>
      <td>56.0</td>
      <td>687</td>
      <td>4.70</td>
      <td>4.74</td>
      <td>3.01</td>
    </tr>
    <tr>
      <th>42827</th>
      <td>42828</td>
      <td>0.51</td>
      <td>Ideal</td>
      <td>E</td>
      <td>SI1</td>
      <td>60.9</td>
      <td>54.0</td>
      <td>1355</td>
      <td>5.17</td>
      <td>5.21</td>
      <td>3.16</td>
    </tr>
    <tr>
      <th>28925</th>
      <td>28926</td>
      <td>0.30</td>
      <td>Ideal</td>
      <td>G</td>
      <td>VVS2</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>684</td>
      <td>4.30</td>
      <td>4.32</td>
      <td>2.65</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-252e1941-271d-49e0-8027-a0df545347e0')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-252e1941-271d-49e0-8027-a0df545347e0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-252e1941-271d-49e0-8027-a0df545347e0');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df.info() # view the data structure and dtypes
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 53940 entries, 0 to 53939
    Data columns (total 11 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   Unnamed: 0  53940 non-null  int64  
     1   carat       53940 non-null  float64
     2   cut         53940 non-null  object 
     3   color       53940 non-null  object 
     4   clarity     53940 non-null  object 
     5   depth       53940 non-null  float64
     6   table       53940 non-null  float64
     7   price       53940 non-null  int64  
     8   x           53940 non-null  float64
     9   y           53940 non-null  float64
     10  z           53940 non-null  float64
    dtypes: float64(6), int64(2), object(3)
    memory usage: 4.5+ MB



```python
df.describe(include = 'all') # include the categoricals in the stats
```





  <div id="df-2d35a60e-6934-47a6-9b69-899d97e4f21f">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940</td>
      <td>53940</td>
      <td>53940</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>7</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Ideal</td>
      <td>G</td>
      <td>SI1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>21551</td>
      <td>11292</td>
      <td>13065</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>26970.500000</td>
      <td>0.797940</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>61.749405</td>
      <td>57.457184</td>
      <td>3932.799722</td>
      <td>5.731157</td>
      <td>5.734526</td>
      <td>3.538734</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15571.281097</td>
      <td>0.474011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.432621</td>
      <td>2.234491</td>
      <td>3989.439738</td>
      <td>1.121761</td>
      <td>1.142135</td>
      <td>0.705699</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.200000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>43.000000</td>
      <td>43.000000</td>
      <td>326.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>13485.750000</td>
      <td>0.400000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>61.000000</td>
      <td>56.000000</td>
      <td>950.000000</td>
      <td>4.710000</td>
      <td>4.720000</td>
      <td>2.910000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>26970.500000</td>
      <td>0.700000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>61.800000</td>
      <td>57.000000</td>
      <td>2401.000000</td>
      <td>5.700000</td>
      <td>5.710000</td>
      <td>3.530000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>40455.250000</td>
      <td>1.040000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>62.500000</td>
      <td>59.000000</td>
      <td>5324.250000</td>
      <td>6.540000</td>
      <td>6.540000</td>
      <td>4.040000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>53940.000000</td>
      <td>5.010000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>79.000000</td>
      <td>95.000000</td>
      <td>18823.000000</td>
      <td>10.740000</td>
      <td>58.900000</td>
      <td>31.800000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2d35a60e-6934-47a6-9b69-899d97e4f21f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2d35a60e-6934-47a6-9b69-899d97e4f21f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2d35a60e-6934-47a6-9b69-899d97e4f21f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




**Notes** <br>
9 variables excluding column one which is an index and target column price and 53940 observations.  Cut, Color and Clarity are categoricals and others are numerical. There seems to be no missing values. 



---



## 2. Explore, Clean & Transform


```python
# Remove column 1 as its not necessary 
df.drop(df.columns[0], axis = 1, inplace = True)
```


```python
# set aside 20% for validation set
validation_set = df.sample(frac = 0.2)
diamond = df[~df.index.isin(validation_set.index)]

# Save the dataset
validation_set.to_csv("dataset/validation_set.csv")
```

#### 2.0 Univariate Analysis on Price


```python
# plot the distribution of Target price and KDE on top as well as Gaussian distribution of sample with the same mean Price for comparison
sample = np.random.normal(loc = diamond.price.mean(),size = diamond.shape[0])
fig, axes = plt.subplots(ncols =2)
plt.rcParams["figure.figsize"] = [20,8]
g = sns.distplot(diamond.price, color = 'orange', ax = axes[0])
g.set_title('Distribution of Diamond Price')
h = sns.distplot(sample, color = 'blue', ax = axes[1])
g.set_title('Normal Gaussian Distribution')
plt.show()
```


    
![png](output_12_0.png)
    



```python
# Calculate Statistics 
print(f"Mean: {diamond.price.mean():.2f}")
print(f"Median: {diamond.price.median():.2f}")
print(f"Standard Deviation: {diamond.price.std():.2f}")
print(f"Skew is: {diamond.price.skew():.2f}")
print(f"Kurtosis is: {diamond.price.kurtosis():.2f}")
```

    Mean: 3928.63
    Median: 2416.00
    Standard Deviation: 3971.33
    Skew is: 1.62
    Kurtosis is: 2.19



```python
# display boxplot of the price
plt.figure(figsize = (15,8))
sns.boxplot(diamond.price, color = 'orange')
plt.title("Diamond Price Boxplot")
plt.show()
```


    
![png](output_14_0.png)
    



```python
# quantify the % of outliers before 1.5IQR + Q3.  
Q1 = diamond.price.quantile(0.25)
Q3 = diamond.price.quantile(0.75)
IQR = Q3 - Q1
high_outlier_threshold = Q3 + (1.5 * IQR)
perc_outliers = diamond.query(f"price > {high_outlier_threshold}").shape[0] / diamond.shape[0]
print(f"Percentage of outliers in dataset is: {perc_outliers *100:.2f}")
```

    Percentage of outliers in dataset is: 6.47


**Notes** <br>
Looking at the distplot the price of diamond is positively skewed at 1.20 with Platykurtic kurtosis at 0.85 when compared to the Gaussian Distribution to the right.  This means that the mean of the data is greater than the median and median is the greater than the mode.  Boxplot also shows that 7% of the data is considered an outlier as they are 1.5IQR from the Q3. In this case, i will remove the extreme outliers at this stage. 


```python
#drop extreme outliers calculated earlier
diamond.drop(diamond.query(f"price > {high_outlier_threshold}").index, inplace =True)
```


```python
# display boxplot of the price with extreme outlier removed
plt.figure(figsize = (15,8))
sns.boxplot(diamond.price, color = 'orange')
plt.title("Diamond Price Boxplot")
plt.show()
```


    
![png](output_18_0.png)
    


#### 2.2 Bivariate Analysis on Price


```python
# list of variables
diamond.columns.tolist()
```




    ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']




```python
# Relationship between target and Carat 
plt.figure(figsize = (15,8))
sns.scatterplot(diamond.carat, diamond.price)
plt.title("Price v Carat")
plt.show()
```


    
![png](output_21_0.png)
    



```python
# Treat by removing outlier greater than 3
diamond = diamond[diamond.carat < 3]

# Relationship between target and Carat 
plt.figure(figsize = (15,8))
sns.scatterplot(diamond.carat, diamond.price, color = 'green')
plt.title("Price v Carat Excluding outliers")
plt.show()
```


    
![png](output_22_0.png)
    



```python
# Relationship between target and Cut 
plt.figure(figsize = (15,8))
sns.violinplot(diamond.cut, diamond.price)
plt.title("Price v Cut")
plt.show()
```


    
![png](output_23_0.png)
    


**Notes**<br>
Above violinplot shows that Ideal cut diamond are selling the most in low prices while Fair cut diamonds are selling all round. Good, premium and Very good seem to have identical price distributions with Premium selling more in the higher end prices.  


```python
# Relationship between target and Color 
plt.figure(figsize = (15,8))
sns.violinplot(diamond.color, diamond.price)
plt.title("Price v Color")
plt.show()
```


    
![png](output_25_0.png)
    


**Notes**<br>
in terms of Color J is the worst color however seems to be selling the least and D is the best follows closely with E to those prices in the lower end in terms of volume.  



```python
# Relationship between target and Clarity 
plt.figure(figsize = (15,8))
sns.violinplot(diamond.clarity, diamond.price)
plt.title("Price v Clarity")
plt.show()
```


    
![png](output_27_0.png)
    



```python
# Relationship between target and Depth
plt.figure(figsize = (15,8))
sns.scatterplot(diamond.depth, diamond.price)
plt.title("Price v Depth")
plt.show()
```


    
![png](output_28_0.png)
    



```python
# Diamond Depth column is very dense with alot of noise so i will create bins and change to categories of 8 bins
diamond['depth_bins']=pd.qcut(diamond.depth, q=8)

plt.figure(figsize = (15,8))
sns.violinplot(diamond.depth_bins, diamond.price)
plt.title("Price v Depth in Bins")
plt.show()
```


    
![png](output_29_0.png)
    



```python
# Relationship between target and Table
plt.figure(figsize = (15,8))
sns.scatterplot(diamond.table, diamond.price)
plt.title("Price v Table")
plt.show()
```


    
![png](output_30_0.png)
    



```python
# Same as above anothe noisy column, i will create 5 bins in this case 
diamond['table_bins']=pd.qcut(diamond.table, q=5)

plt.figure(figsize = (15,8))
sns.violinplot(diamond.table_bins, diamond.price)
plt.title("Price v Table in Bins")
plt.show()
```


    
![png](output_31_0.png)
    



```python
# Relationship between target and x, y, z
plt.rcParams["figure.figsize"] = [15,8]
data = diamond[['x','y','z', 'price']]
g = sns.PairGrid(data)
g.map(plt.scatter)
plt.show()
```


    
![png](output_32_0.png)
    



```python
#drop outlier at 60 on y column and y less than 2
diamond.drop(diamond.query("y>20").index, inplace = True)
diamond.drop(diamond.query("z<2").index, inplace = True)

# Relationship between target and y
plt.figure(figsize = (15,8))
sns.scatterplot(diamond.y, diamond.price)
plt.title("Price v y excluding outlier")
plt.show()
```


    
![png](output_33_0.png)
    



```python
# Relationship between target and z without outliers
plt.figure(figsize = (15,8))
sns.scatterplot(diamond.z, diamond.price)
plt.title("Price v z excluding outlier")
plt.show()
```


    
![png](output_34_0.png)
    



```python
# Factorise the categorical columns to calculate correlation
diamond["table_bins_fact"] = pd.factorize(diamond.table_bins)[0]
diamond["depth_bins_fact"] = pd.factorize(diamond.depth_bins)[0]
diamond["cut_bins_fact"] = pd.factorize(diamond.depth_bins)[0]
diamond["color_bins_fact"] = pd.factorize(diamond.depth_bins)[0]
diamond["clarity_bins_fact"] = pd.factorize(diamond.depth_bins)[0]

#calculate correlation
corr_values = diamond.corr()
plt.rcParams["figure.figsize"] = [15,8]
ax = sns.heatmap(corr_values,annot=True, cmap = 'coolwarm')
ax.set_title('Correlation Matrix')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
```


    
![png](output_35_0.png)
    


**Notes** <br>Based on the above correlation matrix its evident that depth, cut, color and table have a very low corelation with price and hence add no value adde to prediction so i will remove these.  I will also remove y due to multi-collinearity with x thus information will be released by x variable. 


```python
#Save the new training data
train_data = diamond[['price', 'carat', 'x', 'z']]
train_data.to_csv('dataset/train_data.csv')
```



---



## 3. Feature Selection & Preperation

### 3.1 Train & Test split 


```python
X = train_data.drop('price', axis = 1)
y = train_data.price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train.to_csv('X_train.csv')
y_train.to_csv('y_train.csv')
X_test.to_csv('X_test.csv')
y_test.to_csv('y_test.csv')
```



---



## 4. Model Training 

### 4.0 Model 1 - Linear Regression


```python
start_time = time.time()
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(X_train, y_train)
end_time = time.time()
print(f"Linear Regression model accuracy is: {model1.score(X_test, y_test)}")
print(f"Model Execution Time in Seconds:{end_time - start_time:.3f}")
```

    Linear Regression model accuracy is: 0.841303498657225
    Model Execution Time in Seconds:0.023


### 4.1 Model 2 - Support Vector Machines


```python
start_time = time.time()
from sklearn.svm import SVR
model2 = SVR(kernel='linear')
model2.fit(X_train, y_train)
end_time = time.time()
print(f"Linear Support Vector model accuracy is: {model2.score(X_test, y_test)}")
print(f"Model Execution Time in Seconds:{end_time - start_time:.3f}")
```

    Linear Support Vector model accuracy is: 0.7870702787561878
    Model Execution Time in Seconds:71.572


### 4.2 Model 3 - KNN


```python
start_time = time.time()
from sklearn.neighbors import KNeighborsRegressor
model3 = KNeighborsRegressor()
model3.fit(X_train, y_train)
end_time = time.time()
print(f"K-Nearest Neighbour model accuracy is: {model3.score(X_test, y_test)}")
print(f"Model Execution Time in Seconds:{end_time - start_time:.3f}")
```

    K-Nearest Neighbour model accuracy is: 0.8440917291838701
    Model Execution Time in Seconds:0.025


### 4.3 Model 4 - DecisionTree


```python
start_time = time.time()
from sklearn.tree import DecisionTreeRegressor
model4 = DecisionTreeRegressor(random_state = 4)
model4.fit(X_train, y_train)
end_time = time.time()
print(f"DecisionTree model accuracy is: {model4.score(X_test, y_test)}")
print(f"Model Execution Time in Seconds:{end_time - start_time:.3f}")
```

    DecisionTree model accuracy is: 0.7566500495093534
    Model Execution Time in Seconds:0.077


### 4.4 Model 5 - RandomForestTree




```python
start_time = time.time()
from sklearn.ensemble import RandomForestRegressor
model5 = RandomForestRegressor(random_state = 4, n_estimators=100)
model5.fit(X_train, y_train)
end_time = time.time()
print(f"RandomForest model accuracy is: {model5.score(X_test, y_test)}")
print(f"Model Execution Time in Seconds:{end_time - start_time:.3f}")
```

    RandomForest model accuracy is: 0.8320466731778188
    Model Execution Time in Seconds:4.888


### 4.5 Model 6 - XGBoost


```python
start_time = time.time()
import xgboost as xgb
model6 = xgb.XGBRegressor(random_state = 4)
model6.fit(X_train, y_train)
model6.fit(X_train, y_train)
end_time = time.time()
print(f"XGBoost model accuracy is: {model6.score(X_test, y_test)}")
print(f"Model Execution Time in Seconds:{end_time - start_time:.3f}")
```

    [12:10:21] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [12:10:22] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    XGBoost model accuracy is: 0.8641202389980747
    Model Execution Time in Seconds:2.071


### 4.6 Model 7 - Light XGBoost


```python
start_time = time.time()
import lightgbm as lgb
model7 = lgb.LGBMRegressor(random_state = 4)
model7.fit(X_train, y_train)
end_time = time.time()
model7.score(X_test, y_test)
print(f"Light XGBoost model accuracy is: {model7.score(X_test, y_test)}")
print(f"Model Execution Time in Seconds:{end_time - start_time:.3f}")
```

    Light XGBoost model accuracy is: 0.8660241642359542
    Model Execution Time in Seconds:0.285


**Notes** <br>
Light XGBoost Machines perform the best in testset by 2% compared to the nearest model which is KNN at 85% and also best at execution time. 

---



## 5. HyperTunning the Model

### 5.0 Assess the best train and test split %


```python
# test 4 split which produces the highest accuracy 
test = [0.2,0.3,0.4,0.5]
#loop through all test sizes and print the split % and accuracy
for _ in test:
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_, random_state=42)
  model7 = lgb.LGBMRegressor(random_state = 4)
  model7.fit(X_train, y_train)
  print(f"The score for {_} split is: {model7.score(X_test, y_test):.3f}")
```

    The score for 0.2 split is: 0.866
    The score for 0.3 split is: 0.867
    The score for 0.4 split is: 0.866
    The score for 0.5 split is: 0.867



```python
# It looks like there isnt much difference in accuracy. we will stick with 30% split as before.  
```

### 5.1 Hypertune the Model


```python
from sklearn.model_selection import GridSearchCV
model = lgb.LGBMRegressor(random_state = 42)
params = {'learning_rate':[0.01,0.05],'max_depth':[6],
              'min_child_weight': [0.001],
              'silent': [1],
              'subsample': [0.8,1],
              'colsample_bytree': [0.7,0.8,1], 
              'importance_type':['split', 'gain']}

CV = GridSearchCV(model,param_grid=params,scoring='neg_mean_squared_error', cv=5)

#fit the model to training dataset
CV.fit(X_train, y_train)

#get the best parameters of the model
CV.best_params_
```




    {'colsample_bytree': 1,
     'importance_type': 'split',
     'learning_rate': 0.05,
     'max_depth': 6,
     'min_child_weight': 0.001,
     'silent': 1,
     'subsample': 0.8}




```python
start_time = time.time()
model6 = lgb.LGBMRegressor(random_state = 42, importance_type='split',colsample_bytree= 1,learning_rate= 0.05,max_depth= 6,min_child_weight= 0.001, silent= 1,subsample= 0.8)
model6.fit(X_train, y_train)
model6.fit(X_train, y_train)
end_time = time.time()

print(f"Light XGBoost model accuracy is: {model6.score(X_test, y_test):.2f}")
print(f"Model Execution Time in Seconds:{end_time - start_time:.3f}")
```

    Light XGBoost model accuracy is: 0.87
    Model Execution Time in Seconds:0.536


## 6. Feature Importance


```python
#Plot the feature importance from the model - Feature importance show the number of times a feautre was used by the model
plt.xlabel('Feaures')
plt.ylabel('Feature Importance')
plt.title('Feature Importance  from the Light XGBoost Model')
plt.bar(X_train.columns, model6.feature_importances_)
plt.show()
```


    
![png](output_67_0.png)
    


## 7. Prediction on Test & Confidence Interval



```python
# predict the new hypertuned model on test dataset
from sklearn.metrics import mean_squared_error as mse
y_pred = model6.predict(X_test)
RMSE = np.sqrt(mse(y_test, y_pred))
RMSE
```




    996.6380043141057




```python
y_train['prediction'] = y_pred
print(f"Light XGBoost model accuracy is: {model6.score(X_test, y_test):.2f}")
```

    Light XGBoost model accuracy is: 0.87



```python
# calculate the 95% confidence interval of the model
sample_size = X_test.shape[0]
standard_error = RMSE * (np.sqrt(sample_size - 1)/(sample_size -2))
alpha = 1-(95/100)
critical_probability = 1-(alpha/2)
margin_error = (1 +critical_probability) * standard_error
print(f"There is 95% confidence that the model prediction is within {margin_error:.0f} of the price of diamond plus or minus")
```

    There is 95% confidence that the model prediction is within 14 of the price of diamond plus or minus


## 8. Save Model 


```python
import pickle
# now you can save it to a file
with open('model/xgb_model.pkl', 'wb') as f:
    pickle.dump(model6, f)
```

## 9. Prediction on Validation set


```python
# create preprocessing functions for pipeline

# load the model back from pickle file
with open('model/xgb_model.pkl', 'rb') as f:
    model6 = pickle.load(f)

#Load the validation dataset
validation_set = pd.read_csv('dataset/validation_set.csv')

# remove the unwanted columns 
def create_ds(df):
  """Create prediction dataset from original dataset"""
  return df[['price', 'carat', 'x', 'z']]

# split X & Y
def split(df):
  """split X and y"""
  X = df.drop('price', axis = 1)
  y = df.price
  return X, y

validation_set = create_ds(validation_set)
validation_set = validation_set.dropna()
X, y = split(validation_set)

#make prediction 
y_pred_valid = model6.predict(X)
validation_set['predictions'] = y_pred_valid
print(f"Light XGBoost model accuracy on validation data  is: {model6.score(X_test, y_test):.2f}")
```

    Light XGBoost model accuracy on validation data  is: 0.87



```python
# save the dataset with predictions
validation_set.to_csv('dataset/validation_with_prediction.csv')
```

## 10. Conclusion 
Exploratory analytics on the diamond dataset shows us that the best predictors of diamond price from this dataset is the number of carets and the size of the diamond. The rest of the variables such as cut, color, clarity and depth are merely too noisy and complicated for the average buyer of diamonds or we will require large amount of dataset to make sense.  The best performing model on the training data is the Light XGBoost which has been hypertuned to improve the accuracy to 87% final accuracy on validation dataset and the fastest in execution time. 

