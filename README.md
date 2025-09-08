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


### Workflow

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

---

Next, we will analyze the tabulated output of many samples in one experiment:

1. Across growth split conditions
2. Across big vs. small plants
3. Across genotypes

### Diary 

- (09/06/2025): move the repo over and start reformatting code

- **(09/08/2025) goals of the week:**
    1. move all function, including LR model functions
    2. Hirros 13 outputs as a test (since it contains 7 conditions, and 2 genotypes so if it works other should)