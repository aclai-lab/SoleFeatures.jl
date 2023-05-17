# *SoleFeatures.jl* – Feature Selection on Unstructured and Multimodal data

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://aclai-lab.github.io/SoleFeatures.jl/stable)
[![Build Status](https://api.cirrus-ci.com/github/aclai-lab/SoleFeatures.jl.svg?branch=main)](https://cirrus-ci.com/github/aclai-lab/SoleFeatures.jl)
[![Coverage](https://codecov.io/gh/aclai-lab/SoleFeatures.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/aclai-lab/SoleFeatures.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

<!-- [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://aclai-lab.github.io/SoleFeatures.jl/dev) -->

## In a nutshell

SoleFeatures.jl provides tools for filter-based **feature selection** on [*(un)structured* data](https://en.wikipedia.org/wiki/Unstructured_data). At this time, the package provides:
- 3 native feature selection methods, plus a wrapper around python implementations from *scikit-learn* and *scikit-feature*;
- Generalized feature selection methods that also apply to dimensional data (e.g., images or time-series), via a step of window-based flattening;
- Specific methods for time-series feature selection, based on [Catch22.jl](https://github.com/brendanjohnharris/Catch22.jl/);
- An easily extendible codebase, with abstraction layers similar to those of *scikit-learn*.

## Outside the nutshell

In machine learning, *feature selection* is the practice of **reducing the size of a dataset** by selecting only a subset of relevant features. This reduces the time that is necessary for learning models, and it generally leads to models that are more performant (in fact, irrelevant data often misleads machine learning algorithms). Despite the increasing importance of this data processing practice, not only there is not a proper package for it in the Julia environment, but there is also a [lack of implemented algorithms](https://discourse.julialang.org/t/univariate-feature-selection/87414/2) for it.

So, here comes SoleFeatures.jl, which offers a **simple interface for performing feature selection with minimal effort on dataset structures implementing the Table.jl interface**. The package also allows to access partial results of the feature selection processes (e.g., feature importance scores), offering complete control over the whole process. At this time, the package provides 3 native methods based on variance, correlation and statistical hypothesis testing, as well as a wrapper unlocking the majority of *filter-based* methods available in the python libraries *scikit-learn* and *scikit-feature*.

These classical feature selection methods are designed for (un)supervised *structured* datasets, that is, tabular datasets with scalar features; however, an interesting characteristic of this package is the **possibility of using these methods on [*unstructured* data](https://en.wikipedia.org/wiki/Unstructured_data) as well**, for example, with time-series and images. This generalization is possible via a prior *window-based flattening* step, where tailored measures are used to compress the non-scalar components of the data. For example, with time-series data, measures with solid statistical grounds from [Catch22.jl](https://github.com/brendanjohnharris/Catch22.jl/) are used.

All feature selection methods can also be applied to [*multimodal*](https://en.wikipedia.org/wiki/Multimodal_learning) datasets as well. Finally, the package offers an **easily extendible codebase**, with abstraction layers similar to those of *scikit-learn*, and was built as part of [*Sole.jl*](https://github.com/aclai-lab/Sole.jl), a novel programming suite tailored for *unstructured symbolic learning*.

## About

The package is developed by the [ACLAI Lab](https://aclai.unife.it/en/) @ University of Ferrara.

SoleFeatures.jl was built for [*Sole.jl*](https://github.com/aclai-lab/Sole.jl), an open-source framework for *symbolic machine learning*.
