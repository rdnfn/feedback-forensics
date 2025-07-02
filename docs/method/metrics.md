# Metrics 📐

## Annotation Metrics

Our toolkit supports computing the annotation metric below, for comparing two sets of annotations.

### 1. Relevance

Relevance is the proportion of annotations that are valid. We define the *relevance* of one set of annotations over a given set of datapoints as $\texttt{relevance} = n_{\text{valid}}/ n_{\text{total}}$, where $n_{\text{valid}}$ is the number of datapoints with valid votes selecting one response over the other (*response A* or *response B*). This number excludes *tie* (*both*/*neither*) and *invalid* votes. When comparing two sets of annotations ($A$ and $B$), the relevance of annotations $A$ (e.g. personality annotations selecting for more confident responses) over annotations $B$ (e.g. human preference annotations) is the proportion of valid votes over the set of all datapoints included in both annotation sets.

### 2. Cohen's kappa ($\kappa$)

Cohen's kappa is a metric of inter-annotator agreement between two sets of annotations that measures agreement *beyond random chance*. It is defined as

$$\kappa = \frac{p_o - p_e}{1 - p_e},$$

where $p_o$ is the observed proportion of datapoints where annotators agree, and $p_e$ is the proportion of datapoints for which agreement is expected by chance. $p_e$ can be estimated using the observed distribution of labels, as in $p_e=(n_{a_1=A}n_{a_2=A})/N^2 + (n_{a_1=B}n_{a_2=B})/N^2$, where $n_{a_i = X}$ is the number of times annotator $i$ was observed voting for response in position $X$ and $N$ is the total number of observations. For the computation of this metric, we only consider *valid* votes excluding *tie* (*both*/*neither*) and *invalid* votes.

> **Note:** When one of the annotators does not have access to the order of responses (e.g. because they are always shuffled) the expected chance agreement $p_e$ is $0.5$ by design, even if the other annotator is highly biased to one position (e.g. first response). Cohen's kappa in Feedback Forensics is computed under this assumption, given that this randomization is integrated into our personality selecting reference annotators. This kappa version is also used for the computation of the strength metric.

### 3. Strength

Finally, for our specific use-case, we combine *Cohen's kappa* with *relevance* to obtain a measure of *relevant agreement beyond chance*: *strength*. We refer to this metrics as *strength*, defined as

$$\texttt{strength} = \kappa \times \texttt{relevance}.$$

By weighting with relevance, we emphasize agreement that is widely applicable across the dataset. In our setting, this metric indicates whether a personality trait is widely relevant *and* highly correlated with the target annotations. See below for a guide on how to interpret this metric in the two main use-cases.


#### Interpretation of Strength

**Interpretation A:** *Measuring personality traits encouraged by human feedback* (comparing *human* and *personality trait* annotations)
```{figure}  ../img/metrics_strength_interp_casea.png
---
alt: method
width: 100%
align: center
name: fig-method
---
**Interpretation of strength metric** comparing *human* and *personality trait* annotations
```

**Interpretation B:** *measuring personality traits exhibited by model* (comparing *target model* and *personality trait* annotations)
```{figure}  ../img/metrics_strength_interp_caseb.png
---
alt: method
width: 100%
align: center
name: fig-method
---
**Interpretation of strength metric** comparing *target model* and *personality trait* annotations
```

## General Statistics

Feedback Forensics also supports computing the following additional general statistics for each shown data(sub)set of annotations (include in `Overall statistics` table of interface):

- `Number of preference pairs`: Full number of preference pairs included in data(sub)set
- `Prop preferring text_a`: Proportion of datapoints that prefer the first response
- `Avg len text_a`: Average length of first response
- `Avg len text_b`: Average length of second response
- `Avg len pref. text`: Average length of preferred response
- `Avg len rej. text`: Average length of rejected response
- `Prop preferring longer text`: Proportion of datapoints preferring longer text