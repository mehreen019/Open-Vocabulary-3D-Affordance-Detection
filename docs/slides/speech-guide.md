# Presentation Guide: Making OpenAD Robust to Query Phrasing

This guide has two parts for every slide:

- **Understand first** explains the idea in plain language so you know what the slide means.
- **Final speech** is a polished version you can say during the presentation. Learn the ideas rather than memorising every word.

The script is written for roughly a 10 to 12 minute presentation. If you have less time, shorten Slides 4, 8, 9, and 10 before cutting the main problem or results.

## Slide 1: Title

### Understand first

The title contains the whole research question. OpenAD detects affordances, meaning the parts of an object that support an action. It is called open-vocabulary because the action is supplied as text rather than selected from a fixed output layer. Your work asks whether it remains reliable when two phrases have the same meaning but different wording.

Do not explain the full method on this slide. Introduce the team and give the audience one sentence about the problem.

### Final speech

Good morning. We are Mehreen Hossain Chowdhury, Sumaiya Ahmed Rani, and Jannatul Fardus Rakhi from the Department of CSE at Islamic University of Technology. Our project studies how the wording of a language query affects OpenAD, an open-vocabulary model for detecting affordances in 3D objects, and how we can make it more robust to natural phrasing.

## Slide 2: Problem and contribution

### Understand first

An affordance is an action an object supports. Affordance detection is more specific than object classification. It does not only say that an object is a chair; it identifies which points form the sittable area.

OpenAD represents every 3D point using PointNet++ and represents the text using CLIP. It compares these representations to decide which query best describes each point. The important weakness is that OpenAD is trained on labels such as `grasp`, while a real user may ask a complete question. CLIP does not necessarily place the word and the sentence at exactly the same location in its text space.

Your contribution has three parts: reproduce the baseline, measure the phrasing problem, and improve it using paraphrases without changing the architecture.

### Final speech

Affordance detection asks where an action can be performed on an object. For example, it should identify the handle used to grasp a mug, the surface used to sit on a chair, or the blade used to cut with a knife.

OpenAD treats this as an open-vocabulary problem. It compares PointNet++ features from each 3D point with a frozen CLIP embedding of the text query. However, the original model is trained using single words such as “grasp.” A user is more likely to ask, “Where can I grasp this object?” Our first question is whether the model can handle that change in wording.

We reproduce OpenAD, measure this sensitivity, and introduce lightweight prompt-augmented fine-tuning. Rephrasing costs about 12 mIoU points, while updating only 101 thousand parameters recovers more than 90 percent of that loss on unseen wording.

## Slide 3: Dataset and evaluation

### Understand first

The dataset has 18 affordance labels plus a background class called `none`. Every object contains 2,048 points, and the model predicts one of 19 classes for every point.

mIoU means mean intersection over union. For each class, intersection over union compares the overlap between predicted and true points with their combined area. The metric then averages this value over all 19 classes. It is more useful than raw point accuracy here because 43 percent of all points belong to the background class. A model can therefore obtain high accuracy by predicting background everywhere.

The five studied affordances are grasp, contain, sittable, pourable, and cut. Training phrases and held-out phrases are separate. Held-out means the exact text was not used for training. It does not mean the affordance itself is new.

Repeated scores vary slightly because farthest-point sampling begins from a random point. That is why a difference of 0.1 or 0.2 should not be treated as meaningful.

### Final speech

We use the full-shape setting of 3D AffordanceNet as distributed with OpenAD. Each object contains 2,048 points. There are 18 affordance classes and one background class, giving 19 classes in total. The validation split contains 2,285 objects and is strongly imbalanced.

Our main metric is mean intersection over union across all 19 classes. We also report point accuracy and mean class accuracy. We checked our implementation against OpenAD's evaluator and added unit tests.

We study five affordances in three forms: a label, a question, and a description. The fine-tuning and held-out phrase sets were fixed before training and do not overlap. Repeated evaluation varies by about 0.4 mIoU because of random farthest-point sampling, so we treat differences below 0.5 as equivalent.

## Slide 4: Method

### Understand first

There are two branches. The point branch receives the point cloud and produces a 512-dimensional feature for each point. The text branch uses frozen CLIP to produce a 512-dimensional feature for each query. Cosine similarity measures how closely each point matches each query. A softmax turns these similarities into class probabilities.

The orange section is the only part changed during the reported fine-tuning. `fp1` is the last feature-propagation layer of PointNet++. The alignment head maps the final point feature into the same dimensional space as CLIP.

Prompt augmentation means that the wording changes during training. For each of the five affordances, the program randomly selects one training paraphrase at every step. The point cloud and its ground-truth point labels remain exactly the same.

The epoch is selected using canonical words, not held-out phrases. This reduces the risk of choosing a checkpoint because it happened to perform well on the final test wording.

### Final speech

The architecture itself remains unchanged. PointNet++ encodes the 3D object and produces a feature for every point. A frozen CLIP text encoder produces an embedding for every query. The model compares point and text features using cosine similarity, applies a learned scale, and uses a softmax to obtain per-point scores.

Our modification is entirely in the training procedure. At each training step, we randomly sample one paraphrase for each of the five studied affordances. The point labels and loss function remain unchanged. For our main model, we train only the alignment head and the last feature-propagation layer. This is 101,377 parameters, or 5.7 percent of the point network. We train for eight epochs and select the checkpoint using canonical-word validation mIoU, without consulting held-out phrases.

## Slide 5: Experimental setup and baselines

### Understand first

The released pretrained checkpoint is the true baseline. The capacity ladder tests whether more trainable parameters produce more improvement.

The three capacities are nested:

1. alignment head only;
2. alignment head plus `fp1`;
3. alignment head plus all feature-propagation layers.

A fair model comparison uses the same data and evaluation code. When comparing pretrained and fine-tuned models under one phrasing condition, the query list is held constant and the weights differ. When comparing different phrasing conditions for one model, the weights are held constant and only the strings differ.

### Final speech

We compare two query-independent baselines, the released pretrained OpenAD checkpoint, and three versions of our fine-tuned model. The capacity ladder ranges from 67 thousand trainable parameters in the head to 743 thousand in the full decoder.

For a fair comparison, the pretrained and fine-tuned models are evaluated in the same script run with the same objects, ordering, preprocessing, metrics, and query lists. The experiments use PyTorch 2.5 with CUDA 12.1 on one RTX 4060 laptop GPU. One epoch, including validation, takes about 14 minutes, and the supporting code contains 16 unit tests.

## Slide 6: Reproduction

### Understand first

Before trusting new experiments, you must show that your code can reproduce a known result. The most important match is the open-vocabulary result: 14.40 mIoU from your evaluation compared with 14.37 in the paper.

Closed-set means using the original class words. Open-vocabulary here means replacing them with the synonym list used by OpenAD.

### Final speech

Before studying phrasing, we verified that our evaluation pipeline reproduces OpenAD. In the open-vocabulary setting, we obtain 14.40 mIoU compared with the reported 14.37. Point accuracy and mean class accuracy also match closely.

The close reproduction gives us confidence that later differences come from the query wording and fine-tuning rather than an incompatible evaluator.

## Slide 7: Phrasing sensitivity

### Understand first

This is the central problem result. The same pretrained weights are evaluated three times. Only the five query strings change. Canonical words produce 41.5 mIoU. Questions reduce this to 29.6, and descriptions reduce it to 30.3.

Grasp, contain, and pourable fall almost to zero. Cut remains much stronger, which shows that transfer is not uniformly bad. Some text embeddings work well without adaptation.

The correlation analysis compares the spatial pattern of score maps. Even when a question receives a weaker absolute score, it often highlights a similar region. This suggests that PointNet++ still knows where the affordance is, but the final point representation is not aligned strongly enough with the sentence embedding.

Be careful not to claim that correlation proves the backbone is perfect. It is supporting evidence for adapting the alignment layers first.

### Final speech

This table shows the effect of changing only the query wording for the pretrained model. With canonical words, the full 19-class mIoU is 41.5. Replacing five words with held-out questions reduces it to 29.6, and descriptions reduce it to 30.3. Grasp, contain, and pourable fall almost to zero, although cut remains relatively strong.

The score maps still have Pearson correlations between 0.60 and 0.89 across phrasings. This means the response often remains in a similar spatial region even when it loses the final class competition. That observation motivated us to adapt the final point-to-text alignment rather than retrain the whole backbone.

## Slide 8: Main results

### Understand first

`Pre` is the released pretrained model, and `FT` is your fine-tuned model. Seen phrases were eligible for sampling during training. Held-out phrases were not.

The held-out rows are the strongest evidence because they show improvement on exact wordings that were not used during training. Questions improve by 11.21 mIoU, and descriptions improve by 10.48. Canonical words improve slightly rather than becoming worse.

The recovery percentage compares the amount regained with the original gap:

- Question gap: canonical pretrained score minus pretrained question score.
- Recovered amount: fine-tuned question score minus pretrained question score.
- The ratio is about 93 percent.

The same logic gives about 96 percent for descriptions.

Do not say the model generalises to completely new affordances. It generalises to new phrasings of five known affordances.

### Final speech

Prompt-augmented fine-tuning recovers most of the lost performance. On held-out questions, mIoU rises from 29.3 to 40.5, an improvement of 11.2 points. On held-out descriptions, it rises from 30.4 to 40.9, an improvement of 10.5 points. These gains recover 93 and 96 percent of the respective phrasing losses.

The held-out results are only around 1.5 points below the seen phrases. This suggests that the model has learned robustness to a family of wordings rather than memorising the 35 training strings. Canonical performance is also preserved, increasing slightly from 41.3 to 42.0.

## Slide 9: Capacity ablation

### Understand first

An ablation removes or varies one part of a system to identify what matters. Here the variable is how much of the point decoder is trainable.

All three configurations gain approximately 10 to 11 mIoU on held-out phrases. Increasing the trainable parameter count from 67 thousand to 743 thousand does not give a corresponding performance increase. Head plus `fp1` is reported because it has the highest measured held-out scores, but the differences between it and the full decoder are within evaluation variation.

This result supports a practical conclusion: the cheaper update is enough. It also supports, but does not formally prove, the interpretation that the mismatch is concentrated near the point-to-text alignment.

### Final speech

We next ask how much of the network must be trained. The head-only model updates 67 thousand parameters and already improves held-out questions by 10.5 mIoU and descriptions by 10.0. Adding the final propagation layer gives the best measured scores, at 11.2 and 10.5 points of improvement. Training the full decoder, with more than seven times as many parameters as our reported model, does not improve the result further.

The gain is therefore stable across an eleven-fold parameter range. Most of the benefit is already present in the alignment head, so the PointNet++ encoder can remain frozen.

## Slide 10: Training behaviour and synonym transfer

### Understand first

The left table shows that adaptation happens quickly. Validation mIoU peaks at epoch 4. The training loss continues to fall slightly, but validation performance does not continue to improve. This is why the checkpoint from epoch 4 is selected.

The right table uses OpenAD's separate synonym benchmark. `grab` and `take a seat` improve greatly. `accommodate` does not. This means prompt augmentation does not guarantee transfer to every semantically related word.

The 13 unstudied affordances remain almost unchanged on average, from 9.22 to 9.17. This is important because it shows no clear catastrophic forgetting, but it also means there is no evidence of improvement for classes that were never rephrased during training.

### Final speech

Training converges quickly. Canonical validation mIoU peaks at epoch 4 and remains stable afterwards, even though the training loss continues to decrease slightly.

We also evaluate the reported model on OpenAD's synonym benchmark. The total mIoU increases from 14.40 to 18.66. Some synonyms transfer strongly: `grab` improves by 34.6 IoU points, and `take a seat` improves by 31.7. However, `accommodate` does not improve, so transfer remains word-dependent.

Importantly, 97 percent of the overall gain comes from the five affordances used during fine-tuning. The mean score of the 13 unstudied affordances changes only from 9.22 to 9.17, showing no clear degradation but also no broader transfer.

## Slide 11: Per-class synonym results

### Understand first

This figure compares each canonical word with the synonym selected by the OpenAD benchmark. Ten of the 18 synonym queries score below 5 percent IoU. However, `slice` for cut and `hear` for listen perform almost as well as their canonical versions.

The point is not simply that synonyms are difficult. The point is that performance varies greatly by the particular word. Open-vocabulary ability is therefore uneven and should be tested rather than assumed.

### Final speech

This per-class view shows why the average synonym score is low. Ten of the 18 synonyms produce less than 5 percent IoU. For example, some replacements almost completely lose their original class. In contrast, `slice` for cut and `hear` for listen nearly match their canonical words.

Open-vocabulary transfer is therefore highly non-uniform. Whether a query works depends not only on its meaning, but also on how that wording is represented relative to the learned point features.

## Slide 12: Qualitative results and demo

### Understand first

The three columns in each example are ground truth, pretrained map, and fine-tuned map. These are binary-query heat maps against `none`, not the full 19-class segmentation used to calculate mIoU. They show confidence, so do not interpret every coloured point as a final class prediction.

The chair and bowl are clear successes. The knife response becomes stronger. The bag is a failure for both models. Possible explanations include very few bags in the validation set, variation in handle shapes, or a difficult query embedding. These are hypotheses, not proven causes.

The demo allows a user to test different queries and compare the heat maps. Low-confidence warnings matter because a weak response could mean either that the affordance is absent or that the wording was not understood.

### Final speech

These examples compare the ground truth, pretrained score map, and fine-tuned score map for held-out questions. The chair and bowl show clear improvements, and the knife has a stronger peak near the cutting region. The bag remains a failure for both models. This may be related to the small number of bags and the varied geometry of their grasp regions, although further analysis is needed.

We also built a Gradio and Plotly demonstration. Users can choose an object and model, enter a query, inspect a rotatable 3D heat map, compare two phrasings, and see when the model has low confidence.

## Slide 13: Conclusion

### Understand first

The conclusion should answer three questions:

1. What problem did you find? Natural phrasing reduces OpenAD by about 12 mIoU.
2. What did you do? You sampled paraphrases during lightweight fine-tuning.
3. What happened? You recovered over 90 percent of the loss without hurting canonical performance.

Then be honest about scope. The experiment covers five affordances, ten held-out sentence-style phrases, and one training seed. The results are promising but not a universal solution to language variation.

### Final speech

To conclude, we first reproduced OpenAD's open-vocabulary result, obtaining 14.40 mIoU compared with the reported 14.37. We then showed that natural rephrasing of five affordances reduces performance by about 12 mIoU points.

Prompt-augmented fine-tuning updates only 101 thousand parameters and recovers between 93 and 96 percent of this loss on held-out wording. It preserves canonical-word performance and improves the synonym benchmark from 14.40 to 18.66. The capacity study shows that the alignment head provides most of the gain, supporting our view that the main weakness lies in point-to-text alignment rather than geometric feature extraction.

The current study is limited to five affordances, ten held-out phrases, and one seed. Future work should include more varied prompts, additional seeds, synonym evaluation at every unfreezing depth, and consistency-based training objectives.

## Slide 14: Questions

### Understand first

Pause after thanking the audience. Keep the repository link visible. Do not immediately fill the silence.

For questions, return to the relevant slide if useful. If you do not know an answer, distinguish what your experiment demonstrates from what you suspect.

### Final speech

Thank you for listening. We welcome your questions.

## Short answers to likely questions

### Why did you choose only five affordances?

We chose a controlled subset that includes different object regions and different baseline behaviours. This made the experiment feasible while allowing us to test the method clearly. Extending the protocol to all affordances is an important next step.

### Why did you keep CLIP frozen?

The training set contains very few distinct text strings. Training the text encoder could overfit those strings and distort CLIP's broader semantic structure. Keeping it frozen also isolates whether the point-side alignment can be improved.

### Why use mIoU instead of accuracy?

The dataset is imbalanced and contains many background points. A model that predicts only background gets 43 percent accuracy but just 2.27 mIoU. mIoU gives equal weight to every class and better reflects affordance performance.

### Does held-out mean a new affordance?

No. It means a new wording of a known affordance. The experiment measures generalisation across phrasing, not zero-shot learning of new actions.

### Did you prove that the backbone is not the problem?

No. The correlated score maps and the capacity ablation suggest that useful spatial information already exists in the frozen features. They support alignment as the main intervention point, but a direct feature-space analysis would be needed for a stronger causal claim.

### Why report head plus fp1 if the head alone nearly matches it?

Head plus fp1 achieved the highest held-out scores in the capacity study, so we use it as the reported model. We report all three capacities to make that selection transparent, and the main conclusion is the same for every configuration.

### Does the method generalise to affordances that were not fine-tuned?

We found no clear degradation on the 13 unstudied affordances, but we also found no evidence of improvement. The benefit is mainly confined to the affordances exposed to paraphrases during fine-tuning.

### What is the main limitation?

The strongest limitation is scale: five affordances, ten held-out sentence-style phrases, and one training seed. More phrases, more classes, and repeated training runs are needed before making a broad claim about natural-language robustness.
