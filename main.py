from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
# Load breast cancer dataset
X = load_breast_cancer().data
y = load_breast_cancer().target
"""
Scree plot method: This method involves plotting the explained variance ratio against the number of components and choosing the number of components that fall before the "elbow" or the point where the explained variance ratio starts to level off.
"""
# Assuming X is your data matrix
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

"""
Kaiser's rule method: This method suggests keeping all components with an eigenvalue greater than 1.
"""
# Assuming X is your data matrix
pca = PCA().fit(X)
n_components = np.sum(pca.explained_variance_ > 1)
print(f"Kaiser's rule method: {n_components}")
# Kaiser's rule method: 7
"""
Percentage of explained variance method: This method involves choosing the number of components that explain a certain percentage of the variance, such as 95%.
"""

pca = PCA().fit(X)
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print(f"Percentage of explained variance method: {n_components}")
# Percentage of explained variance method: 1


"""
Cross-validation method: This method involves using cross-validation to choose the number of components that result in the best performance on a specific task.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

pca = PCA()
lr = LogisticRegression()
pipe = Pipeline(steps=[('pca', pca), ('logistic', lr)])
param_grid = {
    'pca__n_components': list(range(2, X.shape[1]+1))
}
search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
search.fit(X, y)
n_components = search.best_params_['pca__n_components']
print(f"Cross-validation method: {n_components}")
# Cross-validation method: 12
