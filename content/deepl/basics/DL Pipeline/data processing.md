---
title: Data Processing
---

# Pandas
pandas is the standard tool across research and industry for working with tabular data. 

And in python we will do `import pandas as pd`, and read data as `df = pd.read_csv('data.csv')`.
- Arrange data in a **tabular format**.
- **Extract useful information** filtered by specific conditions.
- Apply numerical operations **using NumPy** to our data.
- **Perform vectorized computations** to speed up our analysis.

## DataFrame

When using pandas, the data is stored in a **DataFrame**, which is a tabular data structure. It is similar to a spreadsheet or a SQL table, but it can also be used to store more complex data types, such as strings or lists.
|   | Landmark         | Type        | Height | Year Built |
|---|------------------|-------------|--------|------------|
| 0 | Sather Gate      | Gate        | 30     | 1910.0     |
| 1 | Campanile        | Tower       | 307    | 1914.0     |
| 2 | Doe Library      | Library     | 80     | 1911.0     |
| 3 | Memorial Glade   | Open Space  | 0      | NaN        |
| 4 | Sproul Plaza     | Plaza       | 0      | 1962.0     |


## Common Operations

1. Create a DataFrame
   ```python
   pd.dataFrame(data)
   pd.read_csv("xxx.csv", index_col='yy')
   ```

2. Quick review
   ```python
   df.head()
   df.tail()
   df.sample()

   df.info()  # a summary including the number of rows, columns, data types, and memory usage. 
   df.describe()  # statistics like mean, standard deviation, and percentiles.
   
   df.shape()  # the number of rows and columns.
   df.zize()  # the total number of elements

   df.value_counts['x']()  # unique values in a column.
   df.unique['x']()  # an array of unique values in a column.
   ```

3. Extraction
   ```python
   df[1:]  # rows 1 to the end
   df['x', 'y']  # columns x and y

   df.loc[[1, 2]]  # by label, inclusive
   df.loc[:, 'x']

   df.iloc[[1,2]]  # by position, exclusive
   df.iloc[1:, [2,4]]

   df[df['x'] > 0]  # filter
   df[df['x'] > 0 and df['y'] < 0]
   df[df['x'].isin(['x1', 'x2'])]
   df.loc[df['x'] > 0]
   ```

4. Modification
   ```python
   df['x'] = [0, 2, 1]
   df['y'] = df['x'] * 0.5

   df.drop(columns=['x'], inplace=True)  # default False
   df.drop(index=[0, 1], inplace=True)

   df.sort_values(by='x', ascending=True)  # default True
   df.sort_values(by=['x', 'y'], ascending=[True, False])

   df.isnull()  # find missing values
   df.fillna(0)  # fill
   df.dropna()
   ```

5. Aggregation
   - Basic: `.sum()`, `.mean()`, `.median()`, `.min()`, `.max()`, `.count()`, `.nunique()`, `prod()`
   - Statistical: `.var()`, `.std()`, `sem()`(standard error of the mean), `.skew()`(skewness), `.kurt()`(kurtosis)
   - Logical: `.all()`, `.any()`
   - Indexing: `.first()`, `.last()`, `.idxmin()`, `.idxmax()`

   - In-Built agg: `sum()`, `max()`, `min()`
   - Numpy agg: `np.sum()`, `np.max()`, `np.min()`, `np.mean()`
   - Pandas agg: `.agg('sum','mean', 'min', 'max', 'first', 'last')`

   ```python
   df.groupby('x').agg('sum')
   ```

6. Join
   ```python
   # 1. Inner join
   df1.join(df2.set_index('x1'), on='x2', how='inner')
   df1.merge(df2, how='inner', left_on='x1', right_on='x2')
   # 2. Outer join
   df1.merge(df2, how='outer', left_on='x1', right_on='x2')
   ```

   | inner | outer | left | right |
   | -- | -- | -- | -- |
   | only both | all both| all left | all right|



# Visualization



# Pytorch


