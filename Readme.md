# Housing Sales Prediction

## Notes

1. See **kNN-timeLeakageModel.py** to see the k-NN model implementation that avoids time leakage.

2. The Relative Median Absolute Error (RMAE) cross-validation score where $cv = 6$ is $0.279 (+/- 0.138)$.

3. We can find a heuristically optimal number $k$ of nearest neighbors, based on Relative Median Absolute Error (RMAE), by using cross-validation. We would have to look at 25 values for $k$ (i.e. $k = 3, \dots, 28$) and perform cross-validation for each value independently. After inspecting the RMAE cross-validation scores for varying $k$, our optimal $k$ value would be the one with the lowest RMAE cross-validation score.

4. The majority of the houses in our dataset are located in the geographical region of Oklahoma and Kansas. As a result, our dataset has a spatial imbalance issue which our K Nearest Neighbors model is not particularly robust to. Consequently, the price predictions of houses outside this geographical region will not be as accurate, which results in much larger relative absolute errors in comparison to the relative absolute errors of price predictions of houses inside this geographical region.

5. Our implementation of this method is computationally expensive. We can implement one effective technique known as preprocessing to speed up $k$-NN classification while the maintaining the level of accuracy. This preprocessing technique would filter a large portion of data instances which are unlikely to match against the unknown pattern. This again accelerates the classification procedure considerably, especially in cases where the dimensionality of the feature space is high. 

6. Since our k nearest neighbor model is computationally intensive, we need to speed up the program with parallelization so that we can productionize it. Specifically, we need to parallelize the getNeighbors method in our $k$-NearestNeighborRegressor class by implementing a a parallelized mergesort algorithm to find the closest neighbors. Furthermore, we need to parallelize the getPredictions method so that our production model can have low latency prediction capabilities. 