```@meta
CurrentModule = SoleFeatures
```

# Univariate Filters

Univariate filter methods evaluate each feature independently based on its statistical relationship with the target variable. These methods are computationally efficient and provide interpretable feature rankings.

## Overview

Univariate filters assess features individually without considering feature interactions. They are particularly useful for:

- **Fast feature selection** on high-dimensional datasets
- **Initial feature screening** before applying more complex methods
- **Interpretable feature rankings** based on statistical tests
- **Handling large feature spaces** where multivariate methods are computationally prohibitive

## Available Filters

```@docs
Chi2Filter
FisherScoreFilter
```

#### Core Functions

```@docs
chi2
fisher_score
```
