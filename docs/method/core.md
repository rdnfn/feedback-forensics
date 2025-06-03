# Method overview ⚙️

This section describes how Feedback Forensics measures personality.

## Measuring personality in feedback

To measure personality traits encouraged by pairwise human or AI feedback, we add additional *personality* annotations. Datapoints in such datasets typically consist of a *prompt*, two *model responses* and an *annotation* selecting the "better" response. We collect additional annotations indicating whether *one response exhibits any of the tested personality traits more*. AI annotators (LLM-as-a-Judge) are used for these annotations. Then, we consider how much the *personality-selecting* agree with the *original* annotations for each tested trait. High agreement indicates that the feedback encourages a certain trait.

## Measuring personality in models

We track model personality changes by considering models *relative* to each other. For each model, we collect a large number of responses to a given prompt list. Then, for each pair of equivalent responses to the same prompt, we annotate whether one response exhibits any of the tested personality traits *more*. This annotation is done by an AI annotator (LLM-as-a-Judge). Finally, to compute the relative *strength* of personality traits for each model, we consider how much each personality trait uniquely identifies a model across all annotations.