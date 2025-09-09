## Dataset Overview

| Hirros | Transplant Date | Genotype                     | Conditions     |
|--------|-----------------|------------------------------|----------------|
| 5      | Late            | Col-0                        | 0, 1, 2, 3     |
| 8      | Early           | Col-0                        | 0, 1, 2, 3     |
| 10     | Late            | Col-0                        | 0, 1, 2, 3     |
| 11     | Early           | Col-0                        | 0, 1, 2, 3     |
| 12     | Early           | Col-0, nrt2.1/2.2, nrtquad   | 1, 2, 3        |
| 13     | Early           | Col-0, Quad                  | 0–6            |


**Conditions:**

| Condition Code | Split Condition             |
|----------------|-----------------------------|
| 0              | KCl 5mM–KCl 5mM             |
| 1              | KCl 5mM–KNO3 5mM            |
| 2              | KNO3 5mM–KCl 5mM            |
| 3              | KNO3 5mM–KNO3 5mM           |
| 4              | KCl 200µM–KNO3 200µM        |
| 5              | KNO3 200µM–KCl 200µM        |
| 6              | KNO3 200µM–KNO3 200µM       |

## Experimental Sample Counts

#### Hirros 5

| Condition Code | Number of Samples |
|----------------|-------------------|
| 0              | 13                |
| 1              | 5                 |
| 2              | 5                 |
| 3              | 7                 |

#### Hirros 8

| Condition Code | Number of Samples |
|----------------|-------------------|
| 0              | 9                 |
| 1              | 7                 |
| 2              | 6                 |
| 3              | 10                |

#### Hirros 10

| Condition Code | Number of Samples |
|----------------|-------------------|
| 0              | 13                |
| 1              | 14                |
| 2              | 13                |
| 3              | 14                |

#### Hirros 11

| Condition Code | Number of Samples |
|----------------|-------------------|
| 0              | 14                |
| 1              | 12                |
| 2              | 13                |
| 3              | 14                |

#### Hirros 12

| Condition Code | Col-0 | Double | Quad | Total |
|----------------|-------|--------|------|-------|
| 1              | 7     | 7      | 6    | 20    |
| 2              | 7     | 8      | 5    | 20    |
| 3              | 9     | 11     | 11   | 31    |
| **Total**      | 23    | 26     | 22   | 71    |

#### Hirros 13

| Condition Code | Col-0 | Quad | Total |
|----------------|-------|------|-------|
| 0              | 10    | 5    | 15    |
| 1              | 4     | 6    | 10    |
| 2              | 4     | 7    | 11    |
| 3              | 8     | 8    | 16    |
| 4              | 4     | 4    | 8     |
| 5              | 4     | 5    | 9     |
| 6              | 9     | 10   | 19    |
| **Total**      | 43    | 45   | 88    |

----

## Workflow

We first need to extract from the root object:

1. **Temporal growth of root system architecture**
   - Primary length over time
   - Lateral length over time (different stage laterals)
   - Lateral count over time
   - Normalized lateral length over time by primary length
   - Mean lateral length over time

2. **Spatial growth of root system architecture**
   - Alpha complex of the RSA with different radii (occupied space)
   - Shape description of occupied space over time, e.g., centroid depth *(not yet done)*

3. **Individual lateral root growth profile**
   - Temporal
   - Spatial *(preliminary: speed, angles; more work needed)*


Next, we will analyze the tabulated output of many samples in one experiment:

1. Across growth split conditions
2. Across big vs. small plants
3. Across genotypes

### Diary 

- (09/06/2025): move the repo over and start reformatting code

- **(09/08/2025) goals of the week:**
    1. move all function, including LR model functions
    2. Hirros 13 outputs as a test (since it contains 7 conditions, and 2 genotypes so if it works other should)

- (09/09/2025): outoput basic csvs and visualizations for all Hirros
