# Unified Project Plan: Interactive 3D Affordance Detection

## 1. What We Are Building

We are building one system for both Pattern Recognition (PR) and Human-Computer Interaction (HCI).

The user selects a 3D object and enters an action in natural language, such as:

- `grasp`
- `Where should I hold this?`
- `Show me the part used for holding`

The system processes the object's 3D point cloud and highlights the points that support the requested action. The user can rotate the object, inspect the prediction, view a confidence summary, change the wording, and compare the new result.

We start from the pretrained **OpenAD PointNet++ model**. We do not claim to have invented OpenAD. Our project contribution is to:

1. Reproduce and evaluate the pretrained model.
2. Fine-tune it to handle varied natural-language phrasing more consistently.
3. Compare the original and fine-tuned models quantitatively and visually.
4. Build an interactive, human-centered interface around the model.
5. Study whether the interface helps users inspect, question, and refine imperfect AI predictions.

## 2. Main Research Question

> Can lightweight prompt-augmented fine-tuning make a pretrained open-vocabulary 3D affordance detector more robust to natural user phrasing, while an interactive interface helps users understand and control its predictions?

We answer this through four comparisons:

1. Original pretrained OpenAD versus our fine-tuned model.
2. Short affordance labels versus natural questions and descriptions.
3. Seen training phrases versus held-out phrasings not used during fine-tuning.
4. Successful predictions versus ambiguous and failed predictions.

## 3. Technical Approach

```text
3D point cloud
      |
      v
PointNet++ visual encoder ---- per-point visual features
                                      |
Natural-language query                |
      |                               |
      v                               v
Frozen CLIP text encoder ---- visual-language similarity
                                      |
                                      v
                         affordance score for each point
                                      |
                                      v
                         interactive colored 3D heatmap
```

The reference OpenAD implementation uses:

- PointNet++ for hierarchical point-cloud feature extraction.
- Frozen CLIP ViT-B/32 for text embeddings.
- Visual-language alignment rather than a fixed closed-set classifier.
- Weighted negative log-likelihood for training.
- Adam with a published initial learning rate of `1e-3`.
- Full-shape point clouds sampled to 2,048 points.

We must verify these details against the downloaded configuration and source code before placing them in the final reports.

## 4. Complete Project Flow

### Step 1: Freeze the scope and experiment

Before coding, commit to the following scope:

- Use the official OpenAD PointNet++ full-shape checkpoint.
- Use the official 3D AffordanceNet/OpenAD data and split.
- Make mIoU the primary metric.
- Fine-tune from the released checkpoint; do not train from scratch.
- Change language supervision through prompt augmentation.
- Build one interface that serves as both the PR demonstration and HCI system.
- Treat partial-view evaluation, alternative backbones, and embedding visualizations as optional work only.

The final claim is an application and targeted adaptation of OpenAD, not a new architecture or a state-of-the-art result.

**Completion check:** every team member can explain the project and contribution in 30 seconds.

### Step 2: Set up the environment and obtain the assets

1. Create a clean Python environment.
2. Install a compatible PyTorch build and OpenAD dependencies.
3. Confirm whether the GPU is visible to PyTorch.
4. Download the official full-shape dataset.
5. Download the pretrained PointNet++ full-shape checkpoint.
6. Configure dataset and checkpoint paths without personal absolute paths.
7. Record the Python, PyTorch, CUDA, GPU, dependency, dataset, and checkpoint versions.

The dataset and checkpoints must not be committed to Git. The README will provide their official download links and expected locations.

**Time limit:** four hours. If the local environment remains blocked, move inference and training to Kaggle or Colab instead of repeatedly rebuilding CUDA.

**Completion check:** the model and one dataset sample load without error.

### Step 3: Reproduce the pretrained baseline

1. Run the official pretrained checkpoint without changing model behavior.
2. Generate a prediction for one object using its canonical affordance label.
3. Export the point coordinates, ground-truth mask, prediction scores, and colored point cloud.
4. Confirm that the prediction is nonempty and visually plausible.
5. Save the exact command, configuration, logs, and output.
6. Wrap the working path in a simple repository command that accepts:
   - configuration path,
   - checkpoint path,
   - sample or object identifier,
   - prompt,
   - output directory.

This pretrained run is the guaranteed baseline and remains usable even if fine-tuning fails.

**Completion check:** one documented command produces a reusable prediction artifact.

### Step 4: Verify the dataset and evaluation protocol

Read the dataset loader, configuration, and official evaluator. Verify:

- Dataset name and version.
- Object categories and sample counts.
- Affordance vocabulary.
- Training, validation, and test split definitions.
- Training terms and open-vocabulary validation synonyms.
- Point sampling, normalization, label mapping, and ignored labels.
- Metric aggregation and prediction thresholding.

Do not quote counts or split details in the reports until they are verified from the actual data and code.

Use the official full-shape open-vocabulary split. Never use test objects, validation synonyms, or held-out evaluation prompts for training.

If the complete evaluation is too slow, define a fixed subset before examining model results. Record every sample ID and the selection rule, use the same subset for every model and prompt condition, and label all results as subset results.

**Completion check:** the exact evaluation population and data boundaries are written down.

### Step 5: Build the evaluation harness

1. Wrap or reuse the official mIoU implementation.
2. Verify it on one or two samples manually.
3. Save per-sample, per-affordance, and aggregate results to CSV or JSON.
4. Report canonical training-word and synonym/open-vocabulary results separately where supported.
5. Add mAP and AUC only if their definitions and implementations can be verified.
6. Add simple random or majority floors only if they are meaningful under the exact output formulation.

The required comparison is:

- Original pretrained checkpoint.
- Our prompt-augmented fine-tuned checkpoint.

Published OpenAD results may appear as cited reference values, but they must be labeled **not reproduced by us** unless we actually reproduce them.

**Completion check:** the pretrained checkpoint has at least one trustworthy metric on a documented evaluation set.

### Step 6: Define the prompt experiment before training

Select three to five affordances supported by the verified vocabulary. For each affordance, create three kinds of prompts:

1. **Label:** `grasp`
2. **Question:** `Where can I grasp this object?`
3. **Description:** `The part of the object used for holding`

Split the paraphrases into two disjoint groups:

- **Fine-tuning prompts:** phrases that may be used during training.
- **Held-out prompts:** different phrases reserved exclusively for evaluation.

The held-out prompts must be written and frozen before fine-tuning starts. This prevents choosing evaluation wording after seeing the results.

Store the prompt sets in version-controlled configuration files. For every comparison, keep the checkpoint, samples, preprocessing, metric implementation, and thresholds fixed; only the prompt should change.

**Completion check:** the affordances, training prompts, and held-out evaluation prompts are frozen and documented.

### Step 7: Measure the pretrained model's prompt sensitivity

Before fine-tuning:

1. Run the pretrained model with the canonical labels.
2. Run it on the same objects with question prompts.
3. Run it again with description prompts.
4. Measure mIoU for each prompt type where ground truth permits.
5. Measure pairwise overlap or correlation between heatmaps produced for equivalent prompts.
6. Record the proportion of points predicted as positive.
7. Identify stable and unstable examples.

This establishes the problem our modification is intended to address. It also motivates the interface's prompt-comparison and query-refinement features.

**Completion check:** a table and several visual examples demonstrate whether wording changes the pretrained model's behavior.

### Step 8: Fine-tune for prompt robustness

The fine-tuning goal is to make semantically equivalent natural-language queries produce more consistent and accurate point-level predictions.

Use the official training objects and masks, but replace or augment the single affordance word with multiple training paraphrases. The spatial target remains the same because equivalent phrases describe the same affordance region.

Fine-tuning procedure:

1. Start from the official pretrained checkpoint.
2. Keep the CLIP text encoder frozen.
3. Begin with the visual-language alignment/projection layers trainable and the earlier PointNet++ layers frozen if the architecture permits this cleanly.
4. If OpenAD has no separable alignment head, unfreeze only the latest PointNet++ feature layers rather than the entire model.
5. Train using only official training objects and the fine-tuning prompt set.
6. Use a lower learning rate than the original from-scratch recipe.
7. Train for a small fixed budget, initially 5 to 10 epochs.
8. Save checkpoints and validation metrics after each epoch.
9. Use early stopping based only on validation performance.
10. Keep the best checkpoint selected without looking at test results.

The exact trainable parameters, learning rate, epochs, batch size, loss, seed, and prompt-sampling method must be recorded. Before training, inspect the actual module names and parameter groups rather than assuming the model has a conventional classifier head.

This is a targeted adaptation using the same dataset, but different language supervision. It is not presented as learning new object geometry or new ground-truth masks.

**Compute limit:** stop after four to six hours of total fine-tuning work. The pretrained baseline must remain the fallback.

**Completion check:** one fine-tuned checkpoint and its complete training record exist.

### Step 9: Run the controlled model comparison

Evaluate both checkpoints under identical conditions:

| Model | Canonical labels | Fine-tuning prompts | Held-out prompts |
|---|---:|---:|---:|
| Original pretrained OpenAD | Required | Required | Required |
| Prompt-augmented fine-tuned OpenAD | Required | Required | Required |

For each condition, report:

- Primary metric: mIoU.
- Per-affordance mIoU where sample size permits.
- mAP and AUC only if verified.
- Heatmap consistency between equivalent prompts.
- Training time and trainable parameter count.

Interpret all important outcomes:

- Fine-tuning improves held-out phrasing without damaging canonical labels.
- Fine-tuning improves training phrases but not held-out phrases, indicating overfitting.
- Fine-tuning improves seen classes but weakens open-vocabulary behavior.
- Fine-tuning provides no improvement or makes performance worse.

A negative result is acceptable if the protocol is correct and the failure is analyzed honestly.

**Completion check:** one fair before-versus-after results table answers the research question.

### Step 10: Produce qualitative and failure analysis

Select at least six examples from the fixed evaluation set:

- Two clear successes.
- Two partial or ambiguous successes.
- Two failures.
- At least one original-versus-fine-tuned comparison.
- At least one canonical-versus-held-out prompt comparison.
- At least one synonym/open-vocabulary example if supported by the protocol.

For each example, record:

- Object and sample identifier.
- Ground-truth affordance region.
- Exact query.
- Original-model heatmap.
- Fine-tuned-model heatmap.
- Metric or score.
- A short interpretation.
- A plausible failure reason without claiming certainty.

Investigate ambiguity, context-dependent actions, small functional regions, neighboring regions, rare affordances, semantically similar actions, and domain shift.

Use a consistent color scale across comparable heatmaps and include ground truth wherever available.

**Completion check:** the selected examples show both strengths and limitations rather than only attractive outputs.

### Step 11: Design the human-centered interaction

Define the user's goal as: ask how an unfamiliar 3D object can be used, inspect the predicted region, and correct or refine the request when the AI is uncertain or wrong.

Prepare the HCI foundation:

1. Identify users and stakeholders.
2. Define functional and non-functional requirements.
3. Create two personas.
4. Write three scenarios:
   - easy: a familiar object and clear query,
   - moderate: an unfamiliar object or paraphrased query,
   - difficult: an ambiguous query or incorrect result requiring refinement.
5. Create a hierarchical task analysis.
6. Create an affinity diagram from available requirements or formative observations.
7. Build a requirements-to-goals traceability matrix with 8 to 12 rows.
8. Sketch the interaction flow and wireframes.

Important HCAI decisions:

- **Explainability:** the heatmap communicates where the prediction comes from.
- **Uncertainty:** scores and warnings avoid presenting predictions as facts.
- **Human control:** the user can rephrase, compare, reject, or reset a query.
- **Trust:** failure examples and limitations are visible rather than hidden.
- **Accessibility:** controls and text are readable, and the color map is not the sole carrier of critical information.

**Completion check:** every major interface feature maps to a user requirement and a known model behavior.

### Step 12: Build the interactive application

Build the smallest complete interface, preferably using Gradio with Plotly or a compatible 3D point-cloud component.

Required interaction:

1. Select a sample object.
2. Enter or choose a natural-language query.
3. Choose the original or fine-tuned model.
4. Run inference or load a cached prediction.
5. View and rotate the colored 3D point cloud.
6. See the exact query and a concise confidence or score summary.
7. Change the query and compare results.
8. Reset the view.

Required states:

- Loading.
- Successful result.
- Low-confidence or uncertain result.
- Unsupported or invalid query.
- Missing model/data error.

Implement the application around cached predictions first. Add live inference only after the complete interaction works. This keeps the demo usable even without presentation-time GPU access.

**Completion check:** another person can complete the full flow without editing code.

### Step 13: Evaluate the HCI system

Preferred evaluation: a small formative study with approximately 3 to 6 available participants after the working prototype exists.

Use tasks derived from the three scenarios. Possible tasks:

1. Find where to perform a stated action on an object.
2. Decide whether the highlighted region appears reasonable.
3. Reformulate an ambiguous query.
4. Compare two prompt results and choose the more useful one.
5. Respond appropriately to a low-confidence or failed prediction.

Measure:

- Task completion or success.
- Completion time.
- Number of query revisions or interaction steps.
- Errors or assistance required.
- Short usability and trust ratings.
- Participant comments and recurring themes.

Document participants, setting, tasks, procedure, measures, and analysis sufficiently for replication. Obtain consent and avoid collecting unnecessary personal data.

If real participants are not feasible, perform an expert cognitive walkthrough and scenario-based evaluation. Simulated data may demonstrate a planned analysis pipeline, but it must be labeled as simulated in the abstract, method, results, and limitations. Never present simulated responses as human-subject findings.

**Completion check:** the HCI evaluation method and evidence are honest, traceable, and tied to the implemented system.

### Step 14: Analyze the unified findings

Bring the PR and HCI findings together:

1. Determine whether prompt-augmented fine-tuning improved accuracy on held-out phrasing.
2. Determine whether it preserved canonical-label and open-vocabulary performance.
3. Identify which prompt types and affordances remained unstable.
4. Explain how those model behaviors informed query refinement, comparison, uncertainty, and failure disclosure in the interface.
5. Analyze whether users or evaluators understood the heatmap and used the available controls appropriately.
6. State limitations in the model, data, evaluation, interface, and study.

Do not claim that a visually convincing result is correct without ground truth. Do not claim usability or trust improvements unsupported by the HCI evidence.

**Completion check:** every conclusion points to a metric, figure, observation, citation, or explicit limitation.

### Step 15: Organize the repository

Target structure:

```text
.
|-- README.md
|-- docs/
|   |-- plans/
|   |-- report-pr/
|   |-- report-hci/
|   `-- slides/
|-- src/
|   |-- data/
|   |-- inference/
|   |-- training/
|   |-- evaluation/
|   |-- visualization/
|   `-- app/
|-- scripts/
|-- configs/
|   |-- prompts/
|   `-- experiments/
|-- tests/
|-- results/
|   |-- metrics/
|   |-- figures/
|   |-- predictions/
|   `-- checkpoints/
|-- demo/
|-- requirements.txt or environment.yml
`-- _ref_openad/
```

Keep `_ref_openad/` ignored and provide a setup script that clones its pinned upstream commit. Attribute its MIT-licensed source. Do not commit datasets, downloaded checkpoints, secrets, machine-specific paths, or temporary files.

The README must explain:

- What the project does.
- What belongs to OpenAD and what we added.
- Environment setup.
- Dataset and checkpoint setup.
- Baseline inference.
- Fine-tuning.
- Evaluation.
- Running the application.
- Reproducing tables and figures.
- Citations and licenses.

**Completion check:** a reviewer can understand and reproduce the documented workflow from the public repository.

### Step 16: Write the reports and presentation

Write while experiments run, but insert only verified final numbers.

For the PR material, emphasize the recognition problem, architecture, conscious design choices, fine-tuning modification, fair experimental protocol, quantitative results, qualitative failures, and demonstration.

For the HCI material, emphasize requirements, personas, scenarios, design rationale, HCAI principles, implementation, evaluation, results, discussion, implications, and limitations.

Use one shared architecture diagram, interface screenshots, qualitative figure set, and system description where appropriate, while changing the analysis for each course.

The PR presentation should contain six main slides:

1. Problem and investigation.
2. Dataset, preprocessing, split, and metrics.
3. Architecture and prompt-augmented fine-tuning.
4. Experimental setup and fair comparisons.
5. Quantitative results and interpretation.
6. Qualitative results, failure cases, and recorded demonstration.

Add backup slides for hyperparameters, metric definitions, training curves, per-affordance results, additional failures, and interface details. Rehearse the main talk to five minutes and never exceed five minutes and thirty seconds.

**Completion check:** both reports compile, the slide deck fits the time limit, and all figures and values match the saved experiment outputs.

### Step 17: Verify, freeze, and submit

1. Run the documented workflow from a clean environment where possible.
2. Compile both reports from scratch.
3. Check every citation, caption, table, split name, metric, and result.
4. Check that pretrained, fine-tuned, reproduced, and cited results are clearly distinguished.
5. Record a short demonstration video; do not depend on live GPU inference.
6. Remove secrets, private paths, datasets, and large checkpoints from Git.
7. Make the GitHub repository public.
8. Verify the repository and shared-drive links in an incognito window.
9. Submit before the stated deadline rather than at the last minute.
10. Keep local copies of the reports, slides, video, results, and repository.

After the final freeze, change only correctness, reproducibility, or presentation-blocking issues.

## 5. Forty-Eight-Hour Execution Schedule

### Hours 0-4

- Complete Steps 1 and 2.
- Start the PR and HCI report templates.
- Reach one loaded model and dataset sample.

### Hours 4-10

- Complete Steps 3 through 5.
- Produce the pretrained baseline prediction and first verified metric.
- Create the prompt configuration.

### Hours 10-18

- Complete Steps 6 through 8.
- Run pretrained prompt sensitivity first.
- Start the bounded fine-tuning run.
- Save every result immediately.

### Hours 18-26

- Complete Steps 9 through 12.
- Produce the comparison table and qualitative figures.
- Build the cached-prediction interface before live inference.

### Hours 26-34

- Complete Steps 13 and 14.
- Run the small formative evaluation or documented fallback.
- Integrate PR and HCI findings.
- Record a first working demo.

### Hours 34-42

- Complete Steps 15 and 16.
- Finish the repository, reports, figures, and slides.
- Record the final demo.

### Hours 42-48

- Complete Step 17.
- Freeze and submit.
- Rehearse the PR presentation repeatedly.

## 6. Go/No-Go Rules

- **Hour 4:** if no sample loads, switch environment or compute platform.
- **Hour 8:** if pretrained inference does not work, stop interface and fine-tuning work until it does.
- **Hour 12:** if the metric remains unclear, report only a verified subset mIoU and document it precisely.
- **Hour 18:** if fine-tuning is broken or too slow, stop it and retain the experiment as an attempted modification; finish the pretrained prompt analysis.
- **Hour 24:** if the live interface is unreliable, use cached predictions.
- **Hour 30:** if participant evaluation is infeasible, perform the expert walkthrough and disclose the limitation.
- **Hour 36:** freeze experiments. Finish reports, repository, demo, and slides.

## 7. Academic Honesty Rules

- Clearly label results as **ours**, **released-checkpoint evaluation**, or **cited from OpenAD**.
- State that our fine-tuning starts from pretrained OpenAD weights.
- Do not call continued or prompt-augmented fine-tuning training from scratch.
- Do not use validation or test examples, synonyms, or held-out prompts for training.
- Report negative results and protocol deviations honestly.
- Cite OpenAD, PointNet++, CLIP, 3D AffordanceNet, reused code, metrics, and HCI frameworks.
- Keep simulated-data disclosure prominent wherever simulated HCI data appears.
- Never describe simulated participants or responses as real.

## 8. Final Project Claim

> We reproduced and evaluated a pretrained OpenAD pipeline for language-guided 3D affordance detection, applied lightweight prompt-augmented fine-tuning to improve robustness to natural phrasing, compared the original and adapted models on canonical and held-out prompts, analyzed successes and failures, and built a human-centered interface for inspecting and refining model predictions. We do not claim a new architecture or state-of-the-art model.

## 9. Deliverables

### Pattern Recognition Deliverables

- Public GitHub repository with setup and reproduction instructions.
- IEEE-style report, unless the team deliberately chooses the permitted Distill-style alternative.
- Five-minute presentation, with a hard maximum of five minutes and thirty seconds.
- Six-slide main deck plus backup slides.
- Architecture/pipeline diagram.
- Documented dataset, preprocessing, split, and metric protocol.
- Reproducible pretrained OpenAD baseline.
- Prompt-augmented fine-tuned checkpoint and training configuration, or an honest failed-experiment account if the bounded run fails.
- Fair pretrained-versus-fine-tuned results table.
- Canonical-versus-natural-versus-held-out prompt comparison.
- At least one verified quantitative metric, with mIoU as the priority.
- Training curves and per-affordance results where available.
- At least six qualitative examples covering successes, partial results, and failures.
- Short recorded system demonstration.
- Clear attribution of pretrained, reproduced, fine-tuned, and cited results.

### HCI Deliverables

- ACM `acmart` report source (`.tex`) and compiled PDF.
- The same working interactive affordance-detection system used for the PR demo.
- User and stakeholder analysis.
- Functional and non-functional requirements.
- Two personas.
- Three scenarios covering easy, moderate, and difficult use.
- Hierarchical task analysis.
- Affinity diagram or documented thematic grouping.
- Requirements-to-goals traceability matrix with approximately 8 to 12 entries.
- Wireframes, interaction flow, and design rationale.
- Human-Centered AI analysis covering explainability, human control, uncertainty, trust, transparency, and accessibility.
- System architecture and feature-to-requirement mapping.
- Formative usability evaluation with real participants where feasible, or a clearly disclosed expert-walkthrough fallback.
- Evaluation tasks, procedure, metrics, results, discussion, design implications, and limitations.
- Prototype demonstration video.
- Shared Google Drive folder containing the reports, video, papers, dataset/software links, and GitHub link, with access verified.
- Prominent simulated-data disclosure in every relevant section if simulated evidence is included.
