# Speaker 2: Short Speech for Slides 6–9

## Slide 6: Reproduction

> Before testing our method, we checked whether our evaluation pipeline was correct. In the closed-set condition, our score is 41.25 mIoU, close to OpenAD's published 42. On the open-vocabulary synonym test, we obtained 14.40, almost identical to the published 14.37. The majority baseline also shows why we use mIoU: predicting background everywhere gives 43 percent accuracy but only 2.27 mIoU. So our evaluation is reliable, and accuracy alone can be misleading.

**Handoff:** “After validating the pipeline, we tested how wording affects the model.”

## Slide 7: Phrasing Sensitivity

> Here we kept the model and data unchanged and only rephrased five queries. For example, `grasp` became “Where should I take hold of this?” or “the surface meant for gripping by hand.” Canonical words give about 41.5 mIoU, while questions and descriptions reduce it to about 30. Grasp, contain, and pourable nearly collapse, although cut remains stronger. This shows that accepting open-vocabulary input does not automatically make the model robust to natural user phrasing.

**Handoff:** “We then used prompt-augmented fine-tuning to address this drop.”

## Slide 8: Main Fine-Tuning Result

> `Pre` is the released pretrained model, and `FT` is our fine-tuned model. Seen phrases could appear during training, while held-out phrases never did. On held-out questions, fine-tuning improves mIoU from 29.27 to 40.48. On held-out descriptions, it improves from 30.40 to 40.88. Canonical performance also remains around 42. So the model improves on new wording without losing its original performance, rather than simply memorizing the exact training sentences.

**Handoff:** “Finally, we tested how much of the network actually needed to be trained.”

## Slide 9: Capacity Ablation

> This ablation compares three amounts of trainable capacity. Training only the 67-thousand-parameter alignment head already improves held-out questions and descriptions by about 10 points. Adding `fp1`, the final point-level feature layer, gives the highest measured result with 101 thousand parameters. Training the full 743-thousand-parameter decoder gives no meaningful extra improvement. Therefore, most of the benefit comes from adapting the final point-to-text mapping, and retraining a much larger part of PointNet++ is unnecessary.

**Final handoff:** “I will now hand over to Rakhi for the transfer analysis, qualitative results, and conclusion.”

## Four-Line Memory Version

1. **Slide 6:** We reproduced OpenAD, so our evaluator is trustworthy.
2. **Slide 7:** Rephrasing five queries drops mIoU from about 41 to 30.
3. **Slide 8:** Fine-tuning raises held-out phrasing back to about 40–41.
4. **Slide 9:** The small alignment head gives almost all the improvement.
