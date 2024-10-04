We have a CSV file of MLS data from California starting in 2023 with 316,908 unique properties.

The goal, as Keith says, is to "find the neighborhood (or two) with the largest spread and just focus on those two neighborhoods and go buy ‘em all."

Plan:

1. Geographical Clustering by lat/long
First, we need to group the properties into clusters based on their location using DBSCAN and K-means algorithms.

2. Attribute Clustering blind to price
We then filter the data to include only detached properties in Northern California and further cluster based on attributes like lot size (sqft) and latitude/longitude. This step ignores price because our goal is to find neighborhoods with the largest price spread. We don't want to compare properties that are too different (e.g., a small condo vs. a large townhouse). Instead, we calculate the price spread across properties with similar attributes that contribute to price.

3. Calculate price spread of each cluster
Now that we have our clusters, we calculate the price spread using the Interquartile Range (IQR). IQR focuses on the middle 50% of data, ignoring extreme outliers. By measuring the difference between the 75th and 25th percentiles, we capture the core price variability without being skewed by unusually high or low prices. This gives us a reliable sense of the typical price spread for each neighborhood cluster, which is what we need to compare similar properties.

4. Rank the clusters by price spread
Finally, we rank the clusters based on their price spread.

Done.
