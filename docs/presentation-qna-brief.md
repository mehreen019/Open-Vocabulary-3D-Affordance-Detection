# Presentation and Q&A Brief

> **Urgent timing note:** the current `docs/slides/speech-guide.md` says it is written for 10 to 12 minutes, but the PR limit is 5 minutes plus at most 30 seconds. Present only the six-slide core described in Section 24. Treat training curves, per-class results, extended ablations, and extra qualitative examples as Q&A backup material.

## 1. The Project in Plain Language

OpenAD can highlight the part of a 3D object associated with an action such as `grasp`, `sit`, or `pour`. It is called open-vocabulary because it compares point features with text embeddings instead of using a conventional fixed classifier.

The practical problem we found is that OpenAD was trained using short class labels, while a real user is likely to type a question or description. For example, the model may understand `grasp` but respond poorly to `Where should I take hold of this?` even though they express the same intent.

We therefore:

1. Reproduced the released OpenAD model and its published open-vocabulary result.
2. Measured how much natural rephrasing changes its predictions.
3. Fine-tuned a small part of the point-side network using paraphrases.
4. Tested on different paraphrases that were never used during fine-tuning.
5. Compared three amounts of trainable model capacity.
6. Analyzed successful and failed predictions.
7. Built an interactive interface for querying and inspecting the model.

## 2. The 60-Second Explanation

> We studied language-guided affordance segmentation in 3D point clouds. The input is a 2,048-point object and a set of text queries; the output is an affordance label for every point. We started from OpenAD, which uses PointNet++ for point features and frozen CLIP for text features. We first reproduced its open-vocabulary benchmark at 14.40 mIoU, very close to the published 14.37. We then found that replacing only five canonical labels with natural questions or descriptions reduced full-vocabulary mIoU from about 41 to about 30. To address this, we randomly sampled paraphrases during fine-tuning while keeping CLIP, the architecture, point labels, and loss unchanged. Our main model updated only the alignment head and final feature-propagation layer, 101,377 parameters or 5.7% of the point network. On held-out questions it improved from 29.27 to 40.48 mIoU, and on held-out descriptions from 30.40 to 40.88. The method recovered over 90% of the phrasing loss without reducing canonical performance. The limitation is that we studied five affordances, ten held-out sentence prompts, and one training seed.

## 3. What Was Existing Work and What Was Ours

### Existing work

- The OpenAD architecture and released source code.
- The released pretrained PointNet++ checkpoint.
- The 3D AffordanceNet dataset.
- PointNet++, CLIP, weighted NLL, and the original OpenAD training method.
- OpenAD's synonym benchmark and published 14.37 open-vocabulary mIoU.

### Our work

- A reproducible inference and evaluation harness for arbitrary query strings.
- Unit-tested mIoU, accuracy, and mean-class-accuracy calculations.
- A frozen prompt protocol separating fine-tuning and held-out phrases.
- Measurement of OpenAD's sensitivity to labels, questions, and descriptions.
- Prompt-augmented fine-tuning starting from the released checkpoint.
- A capacity ablation over three trainable parameter groups.
- Evaluation on held-out questions, held-out descriptions, and OpenAD synonyms.
- Per-class, qualitative, failure, and evaluation-noise analysis.
- A Gradio and Plotly interface for interactive querying and comparison.

Do not say that we invented OpenAD, PointNet++, or open-vocabulary affordance detection. Our contribution is the **prompt-sensitivity study and lightweight prompt-augmented adaptation**.

## 4. The Task and Why It Matters

An affordance is an action an object or object region supports. Examples include:

- a mug handle affords grasping;
- a chair seat affords sitting;
- a knife blade affords cutting;
- a container interior affords containing.

The task is **per-point semantic segmentation conditioned on language**:

- **Input:** point cloud `P` of shape `N x 3` and query strings.
- **Output:** a probability distribution over the query classes for each point.
- **Evaluation prediction:** the highest-scoring query is assigned to each point.
- **Interface heatmap:** one user query competes with `none`, producing a relative per-point query score.

This matters for robotics, assistive systems, and AR because recognizing an object category is not enough. A system often needs to know **where** an action can be performed and must understand how a person naturally expresses that action.

## 5. Dataset and Protocol

We used the full-shape 3D AffordanceNet data distributed with OpenAD.

- 22,949 total object instances, according to the OpenAD paper.
- 23 object categories.
- 18 affordance classes plus the `none` background class.
- 2,048 points per object.
- 2,285 objects in the released validation split.
- No separate test split is included in the distributed data.

The validation set is imbalanced. Table has 799 objects and Chair has 611, together making up approximately 62% of validation objects. Rare categories include Scissors with 6 and Bag with 12.

Preprocessing follows the released loader:

1. Subtract the point-cloud centroid.
2. Divide by the largest distance from the centroid.
3. Place the object inside a unit sphere.
4. Apply no geometric augmentation in our experiment.

Our augmentation is only on the language side.

## 6. How OpenAD Works

### Point branch

PointNet++ processes the unordered 3D points hierarchically:

1. Farthest-point sampling chooses representative centroids.
2. Ball queries collect local neighborhoods at multiple radii.
3. Shared MLPs extract local features.
4. Deeper set-abstraction layers build broader and global context.
5. Feature-propagation layers interpolate features back to all 2,048 points.
6. A `1 x 1` convolution maps each 128-dimensional point feature to 512 dimensions.

### Text branch

Frozen CLIP ViT-B/32 maps every query to a 512-dimensional text embedding.

### Alignment

For each point and text query, OpenAD computes scaled cosine similarity. A softmax across all queries produces the per-point class probabilities.

Conceptually:

```text
point cloud -> PointNet++ -> per-point 512-D embedding
text query  -> frozen CLIP -> text 512-D embedding
                         cosine similarity
                                |
                         softmax over queries
                                |
                     predicted affordance per point
```

The model does not have a conventional fixed 19-output classifier. The text queries act as the class prototypes, which is what allows arbitrary wording at inference time.

## 6A. Exactly What Data We Used and Created

### Did we build the 3D dataset?

No. We downloaded the official 3D AffordanceNet files distributed for OpenAD:

- `full_shape_train_data.pkl`
- `full_shape_val_data.pkl`
- `full_shape_weights.npy`

We used the full-shape files, not the downloaded partial-view files. The original dataset creators produced the objects, point clouds, and point-level annotations. We must credit 3D AffordanceNet for those data.

The full dataset contains 22,949 objects. The released validation file contains 2,285, leaving 20,664 objects in the training split.

### What is one geometric sample?

One sample is one complete 3D object containing:

- a unique shape ID;
- an object category such as Chair, Table, Mug, or Knife;
- 2,048 XYZ points;
- one integer ground-truth label for every point;
- the ordered affordance vocabulary.

Each point label is one of 18 affordances or `none`, so this is a 19-class per-point segmentation problem.

### How did we load and preprocess it?

The official loader reads each object's `coordinate` and `label` arrays from the pickle file. For every object it:

1. Computes the centroid of all points.
2. Subtracts the centroid from every point.
3. Finds the largest point distance from the origin.
4. Divides every coordinate by that distance.
5. Returns normalized XYZ coordinates and the unchanged integer point labels.

This places every object inside a unit sphere. We did not generate new points, change masks, or apply geometric augmentation.

### How many data items did training process?

The fine-tuning loader used the complete official training split:

- 20,664 objects available per epoch;
- batch size 16;
- shuffled order every epoch;
- `drop_last=True`, giving 1,291 complete batches and 20,656 object presentations per epoch;
- 8 epochs for the main head + `fp1` model;
- approximately 165,248 object presentations over those eight epochs.

Because the loader reshuffles, the eight dropped positions need not correspond to the same objects each epoch. Every object contains 2,048 supervised points, so one full non-dropped epoch processes 42,303,488 labeled point positions.

The validation evaluation used all 2,285 objects, or 4,679,680 labeled points, for every reported full-split condition.

### What dataset-like material did we create?

We created the **prompt dataset**, not the point-cloud dataset.

We selected five affordances and manually defined disjoint fine-tuning and held-out paraphrases in `configs/prompts/prompt_sets.yaml`.

For each studied affordance, the fine-tuning pool has:

- 3 label forms;
- 2 question forms;
- 2 description forms;
- 7 phrases total.

Across five affordances, that gives 35 fine-tuning phrases.

For each studied affordance, the held-out pool has:

- 2 label/synonym forms;
- 1 question form;
- 1 description form;
- 4 phrases total.

Across five affordances, that gives 20 held-out phrases. The main natural-sentence evaluation uses the five held-out questions and five held-out descriptions, for ten test sentences. Held-out labels are used separately where applicable, including overlap with OpenAD's official synonym benchmark.

### Did we duplicate each object for every phrase?

No. We did not save a physically expanded dataset with 35 copies of every point cloud.

At each training batch, the code constructs one ordered list of 19 text queries:

- for each of the five studied affordances, randomly sample one of its seven fine-tuning phrases;
- for the remaining 13 affordances, retain the canonical class word;
- retain `none` as the background query.

The same sampled 19-query vocabulary is applied to every object in that batch. The point clouds and label indices stay unchanged. Only the text string occupying the corresponding class position changes.

There are `7^5 = 16,807` possible combinations of the five sampled phrases, although training does not guarantee visiting every combination.

### How many samples were used for each analysis?

- **Training:** all 20,664 official training objects, subject to the eight shuffled positions dropped per epoch.
- **Main pretrained/fine-tuned comparison:** all 2,285 validation objects for every condition.
- **OpenAD synonym reproduction:** all 2,285 validation objects.
- **Random and majority baselines:** all 2,285 validation objects.
- **Prompt-condition mIoU:** all 2,285 validation objects.
- **Heatmap-correlation analysis:** the first 40 validation objects, consisting of 22 doors and 18 clocks; this is supporting evidence only.
- **Qualitative gallery:** six objects. For each chosen category, the first validation object containing the target affordance was used, rather than selecting the visually best prediction.

### What were the data boundaries?

- Training objects were used for gradient updates.
- Validation objects were never used for gradient updates.
- Fine-tuning phrases could be sampled during training.
- Held-out phrases were never sampled during training.
- Canonical validation mIoU selected the best epoch.
- Held-out wording did not select the epoch.

There is no separately released test split. Therefore, using validation for both checkpoint selection and final reporting remains an explicit limitation.

## 7. Why the Phrasing Problem Happens

The point network was originally optimized against short training words such as `grasp`. CLIP gives a question or description a related but different embedding. Even a moderate text-embedding shift can change the relative cosine similarities between a point and all 19 competing queries.

Our heatmap analysis supports this interpretation:

- Canonical and rephrased maps often remain spatially correlated.
- Correlations for the five studied affordances were commonly around 0.60 to 0.89.
- Despite similar spatial ranking, the rephrased affordance can lose the 19-way argmax competition.
- Grasp, contain, and pourable often collapsed almost to zero IoU under natural phrasing.
- Cut was more stable, suggesting that prompt sensitivity varies by concept.

Therefore, useful geometric information may still exist, but the final point-to-text alignment is poorly calibrated for natural wording.

This is supported by the results, but it is not a formal causal proof. A dedicated embedding-space analysis would be needed for that.

## 8. The Prompt Protocol

We studied five affordances:

- `grasp`
- `contain`
- `sittable`
- `pourable`
- `cut`

For every affordance, we created label, question, and description forms. Fine-tuning and held-out strings were disjoint and fixed in configuration before training.

Examples for `grasp`:

- Canonical: `grasp`
- Fine-tuning labels: `grasp`, `grip`, `hold`
- Fine-tuning question: `Where can I grasp this object?`
- Fine-tuning description: `the part used for holding`
- Held-out question: `Where should I take hold of this?`
- Held-out description: `the surface meant for gripping by hand`

Across five affordances, the fine-tuning pool contains 35 phrases: seven per affordance. The main sentence-style test uses ten held-out phrases: one question and one description per affordance.

For any prompt condition, only these five query strings change. The other 13 affordance strings and `none` remain canonical. The point clouds, labels, checkpoint, preprocessing, and evaluator remain fixed.

Held-out here means **new wording of a known affordance**, not a new action class and not unseen geometry.

## 9. What We Fine-Tuned

At every training step, we randomly selected one permitted paraphrase for each of the five studied affordances. The spatial point labels did not change.

Our main configuration trained:

- the alignment head: `conv1`, `bn1`, and `logit_scale`;
- the final PointNet++ feature-propagation layer: `fp1`.

Everything earlier in PointNet++ remained frozen. CLIP remained frozen for every experiment.

Main training configuration:

- Initialization: released OpenAD checkpoint.
- Trainable parameters: 101,377.
- Portion of point network: approximately 5.7%.
- Epochs: 8.
- Learning rate: `1e-4`.
- Weight decay: `1e-4`.
- Batch size: 16.
- Optimizer: Adam.
- Loss: OpenAD's existing weighted negative log-likelihood.
- Seed: 1.
- Checkpoint selection: canonical validation mIoU.
- Best checkpoint: epoch 4, with 42.46 validation mIoU.

We used a lower learning rate and small trainable subset to adapt the alignment without unnecessarily disrupting pretrained geometric features.

## 10. Why CLIP Was Frozen

CLIP contains broad language knowledge learned from a much larger corpus. Our training data expose it to only a small number of affordance phrases. Fine-tuning CLIP would:

- greatly increase trainable parameters and compute;
- risk memorizing the 35 training strings;
- potentially damage its broader language representation;
- make it harder to isolate whether point-side alignment was the problem.

Freezing CLIP makes the experiment cleaner: can the point-side representation be adapted to the existing semantic text space?

## 11. Why mIoU Was the Main Metric

For each class, intersection over union is:

```text
IoU = correctly predicted points / points in prediction or ground truth
```

mIoU averages IoU equally over all 19 classes, including `none`.

Raw accuracy is misleading because `none` is common. The majority baseline predicts `none` for every point and obtains:

- 43.04% point accuracy;
- only 2.27% mIoU.

The random baseline obtains:

- 5.28% point accuracy;
- 1.47% mIoU.

Therefore, mIoU better reflects whether the model actually localizes all affordances rather than exploiting background frequency.

## 12. Result 1: We Reproduced OpenAD

On OpenAD's official synonym query list:

| Result | mIoU |
|---|---:|
| OpenAD paper | 14.37 |
| Our released-checkpoint evaluation | 14.40 |

The 0.03-point difference is much smaller than the observed evaluation variation. This close match is important because it shows that our data order, query order, normalization, checkpoint, and evaluator are compatible with the original benchmark.

This is a reproduction result, not our proposed model's improvement.

## 13. Result 2: Natural Phrasing Hurts the Pretrained Model

With the same pretrained checkpoint and the same data:

| Query condition | Full 19-class mIoU |
|---|---:|
| Canonical words | 41.53 |
| Natural questions | 29.63 |
| Natural descriptions | 30.28 |

Only five of the 19 query strings changed. Nevertheless, mIoU fell by roughly 12 points.

This is the central problem finding: supporting arbitrary text in principle does not guarantee robustness to ordinary user phrasing.

## 14. Result 3: Fine-Tuning Recovered Most of the Loss

The fairest main comparison uses the same evaluation run for both checkpoints:

| Condition | Pretrained | Fine-tuned | Gain |
|---|---:|---:|---:|
| Canonical | 41.28 | 42.02 | +0.74 |
| Seen questions | 29.04 | 42.12 | +13.08 |
| Held-out questions | 29.27 | 40.48 | +11.21 |
| Seen descriptions | 29.64 | 42.37 | +12.73 |
| Held-out descriptions | 30.40 | 40.88 | +10.48 |

For held-out questions, the original phrasing loss was:

```text
41.28 - 29.27 = 12.01 mIoU
```

Fine-tuning recovered:

```text
40.48 - 29.27 = 11.21 mIoU, or about 93% of the loss
```

For held-out descriptions, it recovered about 96% of the loss.

Canonical performance increased slightly instead of being sacrificed. The held-out result matters most because those exact sentences were never sampled during training.

The correct claim is **generalization to held-out phrasings of five known affordances**, not universal natural-language understanding.

## 15. Result 4: A Small Update Was Enough

We compared three amounts of trainable point-network capacity:

| Trainable group | Parameters | Held-out question gain | Held-out description gain |
|---|---:|---:|---:|
| Alignment head | 67,073 | +10.53 | +10.05 |
| Head + final propagation (`fp1`) | 101,377 | +11.21 | +10.48 |
| Full decoder | 743,041 | +10.92 | +10.28 |

All configurations produced similar large improvements. Training over seven times as many parameters did not yield a corresponding gain.

We reported head + `fp1` because it had the highest measured held-out scores. However, differences among fine-tuned configurations are close to the approximately 0.4 mIoU evaluation variation. The stronger conclusion is that **the head already provides most of the benefit**, not that head + `fp1` is definitively superior.

## 16. Result 5: Transfer Was Useful but Localized

On OpenAD's complete 18-synonym benchmark:

| Model | mIoU |
|---|---:|
| Pretrained | 14.40 |
| Fine-tuned head + `fp1` | 18.66 |
| Gain | +4.26 |

Examples:

- `grab` for grasp improved by about 34.6 IoU points.
- `take a seat` for sittable improved by about 31.7 points.
- `pour` improved substantially.
- `accommodate` for contain still failed.

Approximately 97% of the overall gain came from the five affordances included in fine-tuning. The mean score of the other 13 affordances changed from about 9.22 to 9.17, which is effectively stable.

Interpretation:

- The method did not noticeably damage the unstudied classes.
- It also did not produce broad improvement on classes that received no paraphrase exposure.
- Prompt augmentation helps the concepts it covers, but transfer remains word-dependent.

## 17. Per-Affordance Held-Out Results

For the main fine-tuned model:

| Affordance | Held-out question: pre -> fine-tuned | Held-out description: pre -> fine-tuned |
|---|---:|---:|
| Grasp | 0.00 -> 22.19 | 0.32 -> 40.37 |
| Contain | 0.05 -> 42.93 | 0.23 -> 37.75 |
| Sittable | 1.88 -> 58.70 | 21.34 -> 58.22 |
| Pourable | 0.00 -> 42.42 | 0.00 -> 33.88 |
| Cut | 42.35 -> 53.37 | 38.85 -> 54.99 |

The gain is not uniform. Cut was already relatively robust. Grasp remains difficult under the held-out question even after improvement. This is why aggregate results must be accompanied by per-class and qualitative analysis.

## 18. Qualitative Findings

The displayed heatmaps compare ground truth, pretrained response, and fine-tuned response for a user query competing with `none`.

- **Chair/sittable:** clear improvement; the fine-tuned model strongly activates the seat region with a clearer boundary.
- **Bowl/contain:** clear improvement; it recovers the interior but slightly overestimates onto the outer shell.
- **Knife/cut:** partial improvement; the correct blade-edge peak becomes stronger, but the response also rises elsewhere.
- **Mug/pourable:** broad activation; the model recognizes pourability but does not precisely isolate the rim.
- **Bottle/grasp:** a useful peak appears on the labeled region, with elevated background scores.
- **Bag/grasp:** failure; neither model identifies the labeled grasp region.

The qualitative maps use a binary query-versus-`none` setup. They are confidence maps, not the exact 19-way segmentations used for mIoU. Never imply that every colored point is the final multiclass prediction.

## 19. Errors Fine-Tuning Did Not Solve

Some canonical-label weaknesses belong to the underlying detector or dataset rather than prompt wording:

- Pull: approximately 7.1 IoU.
- Lift: approximately 18.5 IoU.
- Openable: approximately 27.7 IoU.
- Pushable: approximately 28.0 IoU.

Low-accuracy object categories include Bag, Bottle, and Hat. Keyboard and Display perform much better.

Our fine-tuning intervention was designed to improve language robustness, not solve geometric ambiguity, category imbalance, annotation limitations, or small-region segmentation.

## 20. The Interactive System

The interface uses Gradio and Plotly. A user can:

1. Select a validation object.
2. Select the pretrained or fine-tuned model.
3. Enter a query or choose an example.
4. Inspect a rotatable colored 3D point cloud.
5. See peak, mean, and fraction-above-threshold score summaries.
6. Compare the current query with the previous query.

The backend caches predictions by object, query, and model. The interface handles:

- normal results;
- low-confidence results, when no point reaches 0.5;
- empty queries;
- inference errors.

Why these features exist:

- Prompt comparison directly addresses the measured phrasing sensitivity.
- Heatmaps expose where the model responds.
- Low-confidence warnings reduce false certainty.
- Query revision preserves human control instead of treating one AI output as final.

The repository does not currently contain finalized participant-study results. Do not claim that the interface was proven to improve usability or trust unless such evidence exists outside this repository and you can cite it accurately.

## 21. Experimental Controls and Fairness

When comparing models under one prompt condition:

- same validation objects;
- same object order;
- same preprocessing;
- same query strings;
- same evaluator;
- only model weights differ.

When comparing prompt conditions for one model:

- same checkpoint;
- same data and evaluation pipeline;
- only five query strings differ.

Held-out sentence prompts were not used in training or epoch selection. Best epoch was chosen by canonical validation mIoU.

However, there is no independent test split. The validation set was used for checkpoint selection and final reporting, and the reported capacity configuration was selected after observing held-out scores. All three capacity results are therefore disclosed. This is a limitation, not something to hide.

## 22. Evaluation Noise and Statistical Scope

Random farthest-point sampling causes small run-to-run differences. Repeated canonical evaluations ranged from 41.19 to 41.53 mIoU, a 0.34-point range. We conservatively treat differences below about 0.5 as practically equivalent.

The fine-tuning gains of approximately 10 to 13 points are far larger than this evaluation variation. However:

- every fine-tuning configuration used only seed 1;
- there are no error bars across training seeds;
- five affordances and ten held-out sentence prompts are a small language sample.

Therefore, the effect is large in this experiment, but its broad statistical generality has not been established.

## 23. Limitations to State Voluntarily

1. We studied five of 18 affordances.
2. The main held-out sentence evaluation contains ten phrases.
3. We used one fine-tuning seed.
4. The released data have validation but no independent test split.
5. Canonical validation mIoU selected the epoch; held-out scores influenced which capacity configuration we highlighted.
6. Evaluation has approximately 0.4 mIoU variation from random sampling.
7. Fine-tuning mainly benefits the affordances exposed to paraphrases.
8. Some heatmaps become broadly activated rather than perfectly localized.
9. The interface score is relative to `none`, not a calibrated real-world probability.
10. No finalized user-study evidence is present in this repository.

Being explicit about these limitations makes the central result more credible.

## 24. Five-Minute Presentation Priority

If time is extremely tight, communicate these points in order:

1. **Problem:** OpenAD sees labels, but people use natural questions.
2. **Validation:** we reproduced 14.40 versus the paper's 14.37.
3. **Finding:** changing five strings drops mIoU from about 41 to about 30.
4. **Method:** sample paraphrases and update only 101k point-side parameters; CLIP stays frozen.
5. **Result:** held-out questions improve 29.27 -> 40.48 and descriptions 30.40 -> 40.88.
6. **Meaning:** over 90% recovery, canonical behavior preserved, larger decoder adds little.
7. **Honesty:** five affordances, ten held-out phrases, one seed, no independent test split.

Do not spend presentation time listing every phrase, layer radius, or per-class value unless asked.

## 25. Likely Q&A

### What exactly is your contribution?

We identified and quantified OpenAD's sensitivity to natural query phrasing, then introduced a lightweight prompt-augmented fine-tuning procedure and evaluated it on disjoint held-out phrases. We also built the evaluation, analysis, and interactive inspection tools around that experiment.

### Did you train the model from scratch?

No. We started from the official pretrained OpenAD checkpoint. Our main adaptation fine-tuned 101,377 point-side parameters for eight epochs. CLIP and most of PointNet++ remained frozen.

### Why fine-tune on the same dataset?

The geometric data and masks remain the same, but the supervision changes from one canonical word per affordance to varied natural-language paraphrases. The research question is language robustness, not learning new object geometry. We evaluate on disjoint phrasings never used for fine-tuning.

### Is this just data augmentation?

It is language-side data augmentation used for targeted model adaptation. That simplicity is intentional. The contribution is showing that a controlled paraphrase intervention can recover most of a large, measured failure without architectural changes.

### Why only five affordances?

They provide a controlled, computationally feasible sample spanning different regions and baseline behaviors. The result demonstrates the method on those affordances, not all language or all actions. Scaling it to all 18 is future work.

### Why keep CLIP frozen?

Our phrase set is too small to safely update a large language encoder. Freezing CLIP preserves its broad semantic space, lowers compute, reduces overfitting risk, and isolates whether the point-side alignment can adapt.

### Why use PointNet++?

Point clouds are unordered and nonuniform. PointNet++ hierarchically samples points and aggregates multi-scale local neighborhoods, allowing it to learn both small functional regions and larger object context. We also used the backbone of the released checkpoint, enabling a fair reproduction.

### Why farthest-point sampling?

It spreads centroids across the object more evenly than purely random sampling, providing spatial coverage for neighborhood feature extraction. Its random starting point causes small evaluation variation, which we measured.

### Why cosine similarity?

It compares the direction of point and text embeddings while reducing dependence on their raw magnitudes. This is appropriate for a shared semantic embedding space and follows OpenAD's design.

### Why weighted NLL?

The point labels are heavily imbalanced, especially because `none` and common affordances dominate. Class weighting prevents frequent classes from overwhelming training. We kept OpenAD's loss unchanged so the experiment isolates prompt augmentation and unfreezing depth.

### Why Adam and learning rate `1e-4`?

Adam is the optimizer used by the reference training approach and works well for adaptation with heterogeneous parameter scales. We used a learning rate ten times lower than the published from-scratch rate to avoid rapidly damaging pretrained features.

### What does 5.7% mean?

The PointNet++-side network contains 1,782,765 parameters excluding frozen CLIP. Our main model updated 101,377 of them: the alignment head and final feature-propagation layer. That is approximately 5.7%.

### What makes the held-out evaluation valid?

Held-out phrases were stored in a disjoint configuration before fine-tuning, and tests prevent overlap with training strings. They were not sampled during training or used to select the epoch. They test phrasing generalization for known affordances.

### Are held-out prompts truly open-vocabulary?

They are unseen strings for known affordance concepts. They demonstrate generalization across phrasing, not entirely unseen action classes. OpenAD's separate synonym benchmark provides the standard open-vocabulary comparison.

### Why did changing only five prompts lower full mIoU so much?

Those five classes often collapsed under rephrasing, and mIoU weights each class equally. In a 19-way softmax, shifting one text embedding also changes its competition with every other query for each point.

### Why can maps correlate while mIoU collapses?

Correlation measures whether points maintain similar relative ranking. mIoU uses the final 19-way class winner. A rephrased class can preserve the rough spatial pattern but have scores low enough to lose the argmax to another class everywhere.

### Did fine-tuning merely memorize the phrases?

Seen prompts score slightly higher, but held-out questions and descriptions still gain 11.21 and 10.48 mIoU. That is evidence of transfer beyond exact string memorization. The small number of held-out prompts means broader language generalization still requires further testing.

### Why select head plus `fp1`?

It produced the highest measured held-out scores. But all three capacities were close, so we report all of them and conclude only that the alignment head provides most of the gain. We do not claim `fp1` is decisively necessary.

### Did you prove the point backbone was already sufficient?

No. Correlated maps and the unfreezing ablation suggest useful spatial information already exists and that alignment is an effective intervention point. A feature-space or causal analysis would be required to prove the backbone is not limiting performance.

### Why is canonical mIoU around 41 while open-vocabulary mIoU is around 14?

The 41 score uses the exact 18 canonical affordance labels seen during original training. The 14 score replaces all 18 with OpenAD's synonym list. They are different query conditions and must not be compared as if they were the same benchmark.

### Why not use accuracy?

Predicting only background already gives 43% accuracy because the data are imbalanced, but it gives only 2.27 mIoU. mIoU gives equal weight to affordance classes and exposes this failure.

### Is the heatmap a calibrated probability?

No. It is a softmax score relative to the competing query, usually `none` in the interface. It is useful for visualization and comparison but should not be interpreted as a calibrated probability of physical success.

### What failed?

The bag grasp example remained a failure, `accommodate` did not improve, and some improved outputs activated overly broad object regions. Canonical weaknesses such as pull and lift also remain because prompt fine-tuning does not solve all geometric or dataset problems.

### Does the method help unstudied affordances?

Not clearly. Their mean synonym IoU remained essentially unchanged, from about 9.22 to 9.17. This is reassuring for preservation but shows that improvement is concentrated in paraphrase-exposed affordances.

### Was there data leakage?

Exact fine-tuning and held-out strings are disjoint, and held-out phrases were not used for epoch selection. However, there is no independent test split: the released validation set was used for checkpoint selection and final evaluation, and held-out results influenced which capacity we highlighted. We disclose this and report all capacity variants.

### Are the gains statistically significant?

We did not perform multi-seed significance testing. The gains of 10 to 13 mIoU are much larger than the measured roughly 0.4 mIoU evaluation variation, but one training seed is insufficient for a formal statistical claim.

### Why did performance peak around epoch 4?

The model starts from strong pretrained weights and updates a small parameter subset, so adaptation occurs quickly. Training loss continued to decline, while canonical validation mIoU stopped improving, indicating that additional epochs were not improving generalization.

### How does the interface connect to the experiment?

The experiment shows that wording changes predictions. The interface therefore lets users revise and compare queries, inspect per-point responses, and see low-confidence warnings. The interaction design directly exposes the measured model behavior rather than hiding it.

### What would you do next?

Run multiple seeds, expand paraphrases to all 18 affordances, add an independent test split, include a canonical-only fine-tuning control, evaluate all capacity variants on the full synonym benchmark, test calibrated uncertainty and threshold-free localization metrics, and conduct a properly documented user study.

## 26. Statements to Avoid

Do not say:

- "We created OpenAD."
- "We trained the model from scratch."
- "The model understands any natural-language action."
- "Held-out means unseen affordance classes."
- "The 40.88 result is on OpenAD's full synonym benchmark."
- "The interface confidence is a calibrated probability."
- "We proved the backbone is not the problem."
- "Head + `fp1` is significantly better than head-only."
- "The system improves all 18 affordances."
- "We proved usability or trust" without actual participant evidence.

Prefer:

- "We adapted a released pretrained OpenAD checkpoint."
- "We generalize to held-out phrasings of five known affordances."
- "The evidence suggests the final alignment is an effective intervention point."
- "The improvement is concentrated in the five studied affordances."
- "The interface supports inspection and query refinement."

## 27. Final Three-Sentence Answer

If you become stuck during Q&A, return to this:

> We reproduced OpenAD and found a practical gap between canonical training labels and natural user wording. We addressed it by sampling paraphrases while fine-tuning only 5.7% of the point-side model, which recovered more than 90% of the loss on held-out questions and descriptions without reducing canonical performance. The result is promising but scoped to five affordances, ten held-out sentence prompts, and one seed.
