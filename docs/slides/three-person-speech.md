# Three-Person Presentation Script

**Target duration:** 5:00 minutes  
**Hard stop:** 5:30 minutes  
**Speaking pace:** approximately 130 words per minute

## Speaker Split

| Speaker | Slides | Topic | Target time |
|---|---|---|---:|
| Mehreen | Title, 1–3 | Problem, data, and method | 0:00–1:40 |
| Sumaiya | 4–7 | Experimental validity, phrasing problem, main result | 1:40–3:20 |
| Rakhi | 8–12 | Ablation, transfer, examples, limitations, conclusion | 3:20–5:00 |

Do not stop for the demo during the five-minute talk. Show the prepared qualitative output and mention that the interactive system is available. Use the actual demo only if the instructors request it during Q&A.

---

## Speaker 1: Mehreen

### Title and Slide 1: Problem and Contribution

**Time: 0:00–0:40**

> Assalamu alaikum. We are Mehreen, Sumaiya, and Rakhi. Our project studies language-guided affordance detection in 3D point clouds.
>
> An affordance describes where an action can be performed, such as sitting on a chair seat or grasping a mug handle. OpenAD detects these regions using PointNet++ and CLIP. But it is trained with short labels such as “grasp,” while a user may ask, “Where should I hold this?” We measure this wording problem and test a lightweight solution.

### Slide 2: Dataset and Evaluation

**Time: 0:40–1:10**

> We used the official full-shape 3D AffordanceNet dataset, not a new point-cloud dataset. It has 20,664 training and 2,285 validation objects. Each object has 2,048 points labeled with 18 affordances or background.
>
> We created 35 fine-tuning paraphrases and a disjoint held-out set. Our main metric is mIoU because ordinary accuracy is misleading when background points are common.

### Slide 3: Method

**Time: 1:10–1:40**

> PointNet++ encodes 3D geometry, frozen CLIP encodes the queries, and scaled cosine similarity assigns every point. We keep this architecture unchanged. During fine-tuning, we randomly replace five labels with equivalent labels, questions, or descriptions. We update only the alignment head and final propagation layer: 101,377 parameters, or 5.7 percent of the point network.
>
> Sumaiya will now explain how we validated and evaluated this modification.

---

## Speaker 2: Sumaiya

### Slide 4: Experimental Setup and Baselines

**Time: 1:40–2:00**

> We compared majority-class and random floors, pretrained OpenAD, and three fine-tuning capacities. Every model comparison uses the same objects, preprocessing, queries, and evaluator; only the weights differ. Our main model trained for eight epochs with Adam at a learning rate of ten to the minus four. Canonical validation mIoU selected the checkpoint, not held-out wording.

### Slide 5: Reproduction

**Time: 2:00–2:20**

> We first reproduced OpenAD’s results. Our closed-set score is 41.25 mIoU versus the paper’s 42.00, and our synonym benchmark is 14.40 versus 14.37. The majority baseline gets 43 percent accuracy but only 2.27 mIoU, showing why mIoU matters. These results validate our evaluation pipeline.

### Slide 6: The Phrasing Problem

**Time: 2:20–2:45**

> Next, we kept the pretrained model fixed and changed only five query strings. Canonical words produce about 41.5 mIoU; questions reduce this to 29.6 and descriptions to 30.3. The heatmaps remain spatially related, but rephrased classes often lose the final competition against other queries.

### Slide 7: Main Result

**Time: 2:45–3:20**

> Fine-tuning substantially improves these results. Held-out questions rise from 29.27 to 40.48, and held-out descriptions from 30.40 to 40.88. These exact phrases were never used in training, while canonical performance remains preserved at about 42 mIoU.
>
> Rakhi will now discuss how much of the network was necessary and where the method still fails.

---

## Speaker 3: Rakhi

### Slide 8: Capacity Ablation

**Time: 3:20–3:40**

> We trained three capacities, from the 67-thousand-parameter head to the 743-thousand-parameter decoder. All recover roughly 10 to 11 mIoU. The larger decoder adds little, suggesting that the final point-to-text alignment provides most of the benefit.

### Slides 9 and 10: Training and Transfer

**Time: 3:40–4:05**

> Validation performance peaks at epoch four. On OpenAD’s synonym benchmark, our model improves from 14.40 to 18.66. But 97 percent of the gain comes from the five trained affordances. The other 13 remain stable, so there is no clear degradation, but also no universal transfer.

### Slide 11: Qualitative Results and Interface

**Time: 4:05–4:30**

> The chair and bowl show clear recovery, while the mug and bottle activate too broadly and the bag remains a failure. Our Gradio and Plotly interface lets users query a rotatable heatmap, compare wording, and see low-confidence warnings.

### Slide 12: Conclusion

**Time: 4:30–5:00**

> In conclusion, natural rephrasing costs OpenAD about 12 mIoU. Fine-tuning only 5.7 percent of the point network improves held-out questions by 11.2 and descriptions by 10.5 mIoU while preserving canonical performance.
>
> Our scope is five affordances, ten held-out sentences, one seed, and no independent test split. This is a promising lightweight improvement, not universal language robustness. Thank you. We welcome your questions.

---

## Handoff Cues

- **Mehreen to Sumaiya:** “Sumaiya will now explain how we validated and evaluated this modification.”
- **Sumaiya to Rakhi:** “Rakhi will now discuss how much of the network was necessary and where the method still fails.”
- Do not repeat greetings or introduce the next speaker again.
- The next speaker should begin immediately while the slide changes.

## Emergency Short Version

If the team is behind time:

- Mehreen skips the exact prompt counts on Slide 2.
- Sumaiya skips the hardware details on Slide 4.
- Rakhi summarizes Slides 9 and 10 in one sentence: “The synonym benchmark improves from 14.40 to 18.66, but the gain is concentrated in the five trained affordances.”
- Never cut the held-out results or limitations.
