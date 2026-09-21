# Speaker 2 Preparation: Slides 6–9

**Your responsibility:** reproduction, phrasing sensitivity, main results, and ablation  
**Target time:** 3.5–4 minutes  
**Main story:** we verified the pipeline, discovered a large phrasing problem, fixed most of it, and showed that a small parameter update was enough.

## Your Four-Slide Story

Remember this sequence:

1. **Can we trust our implementation?** Yes, because we reproduced OpenAD.
2. **What problem did we discover?** Natural wording reduces mIoU by about 12 points.
3. **Did our method solve it?** Held-out questions improve by 11.21 mIoU and descriptions by 10.48.
4. **How much training was necessary?** The small alignment head gave almost all the benefit.

## Full Speech

### Slide 6: Reproduction Validates the Harness

**Target: 45–50 seconds**

> Before testing our modification, we first checked whether our evaluation pipeline could reproduce a known OpenAD result.
>
> In the closed-set condition, where the model receives the original training labels, our released-checkpoint evaluation obtains 41.25 mIoU, close to OpenAD's reported 42.00.
>
> More importantly, on OpenAD's official open-vocabulary synonym benchmark, we obtain 14.40 mIoU compared with the published 14.37. The 0.03 difference is much smaller than the approximately 0.4 mIoU variation caused by random farthest-point sampling.
>
> This close reproduction indicates that our checkpoint, query order, data normalization, and metric implementation are compatible with the original work.
>
> The majority-class baseline always predicts `none`. Because background points are common, it gets 43.04 percent raw accuracy, but only 2.27 mIoU because it fails the affordance classes. This shows why mIoU is the more meaningful metric here.
>
> The random baseline assigns one of the 19 labels to each point uniformly at random. Its 5.28 percent accuracy is approximately the expected chance level of one divided by 19.
>
**What this slide means:** You did not simply assume your evaluator was correct. You checked it against a published anchor before trusting new experiments.

### Slide 7: The Problem — Phrasing Breaks the Pretrained Model

**Target: 50–55 seconds**

> After validating the pipeline, we kept the pretrained model and validation data fixed and changed only five query strings.
>
> With canonical words, such as `grasp` and `contain`, the model obtains about 41.5 mIoU. Replacing those five words with natural questions reduces the full 19-class score to 29.6, while descriptions reduce it to 30.3. That is a loss of approximately 12 mIoU even though only five of the 19 queries changed.
>
> The effect differs across affordances. Grasp, contain, and pourable fall almost to zero, while cut remains comparatively stable.
>
> Interestingly, the point-score maps remain spatially correlated, with correlations between about 0.60 and 0.89. This means the model often still looks in a similar region, but the response becomes too weak to win the final 19-way class competition. That finding motivated us to adapt the final point-to-text alignment rather than retrain the entire geometric backbone.

**What this slide means:** Open-vocabulary capability does not automatically mean robustness to normal user language.

### Slide 8: Main Result — Fine-Tuning Improves Held-Out Phrasing

**Target: 60–65 seconds**

> This table contains our main result. `Pre` is the released pretrained checkpoint, while `FT` is our prompt-augmented model. Seen phrases were eligible for sampling during training; held-out phrases were completely excluded from training and epoch selection.
>
> On held-out questions, the pretrained model obtains 29.27 mIoU and the fine-tuned model obtains 40.48, an improvement of 11.21 points. On held-out descriptions, performance rises from 30.40 to 40.88, a gain of 10.48.
>
> Canonical performance is also preserved, changing from 41.28 to 42.02. Therefore, the model did not gain robustness by sacrificing its original label performance.
>
> The held-out rows are our strongest evidence because the exact test sentences were never used during fine-tuning. However, these are new phrasings of five known affordances, not completely unseen action classes.

**What this slide means:** The method transfers beyond memorized training strings while preserving the original task.

### Slide 9: Ablation — How Much of the Network Must Be Trained?

**Target: 50–55 seconds**

> We next varied how much of the point network was trainable.
>
> Training only the alignment head updates 67 thousand parameters and already improves held-out questions and descriptions by roughly 10 points. Adding the final feature-propagation layer increases this to 101 thousand parameters and gives the highest measured scores. Training the full decoder updates 743 thousand parameters but does not provide a corresponding improvement.
>
> The differences among the three fine-tuned models are close to our evaluation variation, so we do not claim that head plus `fp1` is significantly better than head-only. The defensible conclusion is that the alignment head provides most of the gain, and substantially more trainable capacity is unnecessary.
>
> This supports a lightweight adaptation strategy and suggests that useful spatial features already exist in the pretrained network.
>
> I will now hand over to Rakhi for the transfer analysis, qualitative results, and conclusion.

**What this slide means:** The improvement does not require retraining the full PointNet++ network.

## Compact Version to Memorize

If you lose your place, remember these four lines:

1. **Reproduction:** `14.40` versus published `14.37`; our pipeline is compatible.
2. **Problem:** natural questions/descriptions reduce mIoU from about `41` to `30`.
3. **Solution:** held-out questions reach `40.48`; descriptions reach `40.88`; canonical stays near `42`.
4. **Ablation:** `67k`, `101k`, and `743k` parameters perform similarly; the small head gives most of the gain.

## Numbers You Must Know

| Topic | Number | Meaning |
|---|---:|---|
| Published OpenAD synonym result | 14.37 | Cited from the paper |
| Our released-checkpoint synonym result | 14.40 | Our reproduction |
| Majority accuracy | 43.04 | High because `none` is common |
| Majority mIoU | 2.27 | Shows that it fails actual affordances |
| Random accuracy | 5.28 | Approximately chance over 19 classes |
| Random mIoU | 1.47 | Lower reference floor |
| Pretrained canonical | about 41.3–41.5 | Exact value varies slightly across runs |
| Pretrained held-out question | 29.27 | Same model, five queries rephrased |
| Fine-tuned held-out question | 40.48 | Gain of 11.21 |
| Pretrained held-out description | 30.40 | Before adaptation |
| Fine-tuned held-out description | 40.88 | Gain of 10.48 |
| Fine-tuned canonical | 42.02 | Original performance preserved |
| Head-only capacity | 67,073 | Already gives most of the gain |
| Head + `fp1` capacity | 101,377 | Reported model; approximately 5.7% |
| Full decoder capacity | 743,041 | No meaningful extra gain |
| Evaluation variation | about 0.4 mIoU | Differences below 0.5 treated cautiously |

Do not panic if canonical mIoU appears as 41.25, 41.28, 41.44, or 41.53 on different tables. These came from separate evaluation passes with random farthest-point sampling. State “about 41 to 41.5” unless discussing a specific row.

## Concepts You Need to Understand

### Closed-set versus open-vocabulary evaluation

- **Closed-set/canonical:** all 18 affordances use the original training words.
- **OpenAD synonym benchmark:** all 18 words are replaced with OpenAD's synonyms.
- **Our phrasing experiment:** only five affordances are replaced with questions or descriptions; the other queries stay canonical.

That is why canonical mIoU is about 41, while the complete synonym benchmark is about 14. They are different query conditions.

### Seen versus held-out phrases

- **Seen phrase:** belongs to the pool from which training could sample.
- **Held-out phrase:** stored in a disjoint configuration and never sampled during training.
- Held-out does **not** mean an unseen object or unseen affordance concept.
- It tests whether adaptation transfers to a new way of expressing a known action.

### Why mIoU falls when only five strings change

mIoU gives equal weight to all 19 classes. Several changed classes collapse almost completely. Also, prediction is an argmax across all queries, so weakening one query causes other unchanged classes to win those points.

### Why maps can correlate while IoU collapses

Correlation preserves relative spatial ranking: both maps may peak in the same area. IoU uses the final class decision. If every score becomes weaker, the correct region can lose the argmax even though its internal shape remains similar.

Use this concrete example:

```text
Mug-handle points

With "grasp":
handle scores are high enough to win -> predicted as grasp

With "Where should I take hold of this?":
handle is still the query's highest-scoring region,
but all of its query scores are weaker -> another class wins
```

Pearson correlation compares the shape of the two score maps across the 2,048 points. A positive value such as 0.7 means points scored relatively high by one wording also tend to score relatively high with the other wording. It does not mean the absolute scores are equally strong or that the final labels are correct.

Therefore say:

> “The model still tends to look at the same region, but the natural phrase matches that region less strongly. This suggests the spatial features remain useful and the point-to-text match needs adaptation.”

Do not say that correlation proves the backbone is correct. It is supporting evidence only.

### What the ablation proves

The ablation shows that training more decoder parameters did not meaningfully increase the measured gain. It supports using a small alignment update.

It does **not** prove that the geometric backbone is perfect or never contributes to errors.

## Questions You Are Most Likely to Receive

### Why did you reproduce OpenAD first?

Because a new improvement is not meaningful if the baseline pipeline is incompatible with the original benchmark. Matching 14.40 to 14.37 validates our checkpoint, data processing, query order, and metrics.

### What is the majority-class baseline?

It predicts `none` for every point without examining the object. It needs no training or additional data. Its 43.04% accuracy is misleading because background is common, while its 2.27 mIoU reveals that it fails the affordance classes.

### Why keep this baseline?

It demonstrates why raw point accuracy is not sufficient for this imbalanced segmentation problem and motivates mIoU as the primary metric.

### What is the random baseline?

It assigns one of the 19 labels to each point with equal probability. It does not inspect the object and needs no training or extra data. Its 5.28% accuracy closely matches the expected chance level of `1/19`, or 5.26%.

### Did you evaluate the whole validation set?

Yes. Every main mIoU value uses all 2,285 validation objects, each with 2,048 points. Only the supporting heatmap-correlation analysis used 40 objects.

### Did you build the dataset?

No. We used the official 3D AffordanceNet point clouds and masks distributed with OpenAD. We created the prompt sets: 35 fine-tuning phrases and disjoint held-out phrases.

### Why did you study only five affordances?

The controlled subset made the experiment feasible while covering different functional regions and baseline behaviors. The result is limited to those five; scaling to all 18 is future work.

### Are the held-out prompts new classes?

No. They are new sentences for known affordance concepts. This measures phrasing generalization, not zero-shot transfer to entirely new actions.

### Did the model memorize the fine-tuning phrases?

The seen phrases score highest, but held-out questions gain 11.21 mIoU and held-out descriptions gain 10.48. That shows transfer beyond exact string memorization, although the held-out set is small.

### Why did canonical performance improve slightly?

The canonical labels themselves were included in the fine-tuning phrase pool, and the checkpoint was selected using canonical validation mIoU. The important conclusion is preservation, not that the small increase is necessarily significant.

### Why did you choose head plus `fp1`?

It had the highest measured held-out scores. However, the differences from head-only are close to evaluation noise, so all configurations are reported and we only claim that small alignment-side adaptation is sufficient.

### Did you prove that phrasing is the only problem?

No. Prompt sensitivity is one important problem. The model still has geometric, class-imbalance, and dataset-related failures even with canonical labels.

### Are the gains statistically significant?

We did not run multiple training seeds or a formal significance test. The 10–13 point gains are much larger than the measured 0.4-point evaluation variation, but broader statistical validation requires more seeds.

### Was there data leakage?

Exact held-out phrases never appeared in training and did not select the epoch. However, the released dataset has no separate test split, so validation was used both for canonical checkpoint selection and final reporting. We disclose this limitation.

## Statements to Avoid

Do not say:

- “We achieved 40.88 on the full open-vocabulary synonym benchmark.” That benchmark result is 18.66 after fine-tuning.
- “Held-out means unseen affordance.” It means unseen wording.
- “We proved the backbone is not the problem.” Say the results suggest alignment is an effective intervention point.
- “Head plus `fp1` is significantly better.” The measured differences are close to evaluation noise.
- “Our model understands arbitrary natural language.” The test covers five affordances and ten held-out sentence prompts.
- “The 14.37 result is ours.” It is cited; 14.40 is ours using the released checkpoint.

## Handoff

End with:

> I will now hand over to Rakhi for the transfer analysis, qualitative results, and conclusion.

Then stop speaking immediately and let the next speaker begin.
