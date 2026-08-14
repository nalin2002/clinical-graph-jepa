# Fawkes Poster Q&A Prep

Audience: investors, researchers, engineers, clinical/product leaders, and general conference attendees at the WM@Booth poster session.

Use this as a rehearsal sheet. The answers are intentionally spoken-answer length: enough to be credible at the poster without over-explaining.

## Core One-Liners

**Q: What is Fawkes in one sentence?**  
A: Fawkes is a self-supervised graph model that learns patient-state representations from per-admission clinical knowledge graphs, then uses those representations to recover clinically meaningful missing or uncertain relations.

**Q: What problem are you solving?**  
A: Clinical notes contain relationships that structured tables often miss, but automatically extracted graphs can be noisy or incomplete. We treat the graph as a draft patient state and train the model to infer which relationships are likely supported by the surrounding context.

**Q: Why should I care about graph revision instead of another clinical predictor?**  
A: Most clinical AI jumps straight to an outcome label. This work improves the intermediate representation: the patient graph itself. A cleaner, more complete graph can support downstream reasoning, retrieval, cohorting, and decision-support workflows more transparently.

**Q: What is the headline result?**  
A: The paper Fawkes entity-note checkpoint reports leave-one-out edge-recovery MRR of 0.418653 on the seeded 400-admission test split, with Hits@10 of 0.863576 over 8,283 queries.

**Q: What does MRR mean here?**  
A: For each hidden true edge, the model ranks the correct target among type-compatible alternatives. MRR rewards ranking the true target near the top; 1.0 would mean the hidden target is always ranked first.

**Q: Is this making clinical decisions?**  
A: No. This is a research system for improving structured patient representations. It does not diagnose, recommend treatment, or make autonomous clinical decisions.

**Q: What is the simplest demo explanation?**  
A: Remove a true relation like "medication managed for diagnosis," show the rest of the patient's graph, and ask whether the model ranks the missing target diagnosis above plausible alternatives.

## Method And Novelty

**Q: What is JEPA?**  
A: JEPA stands for Joint Embedding Predictive Architecture. Instead of reconstructing raw text or graph tokens, the model predicts the latent representation of missing graph state from observed context.

**Q: Why use JEPA for clinical graphs?**  
A: Clinical graphs are sparse, noisy, and relational. Predicting latent structure encourages the model to learn patient-state regularities without needing a manually labeled outcome for every admission.

**Q: What exactly gets masked during Fawkes pretraining?**  
A: In the Fawkes paper implementation, 40% of nodes are hidden. The context encoder sees the visible subgraph, while the EMA target encoder sees the full graph and supplies the latent target for hidden nodes.

**Q: How is the model trained after JEPA pretraining?**  
A: The graph encoder is frozen, then a DistMult relation readout is trained with InfoNCE to rank the true target above eight same-type negatives.

**Q: Why freeze the encoder for the readout stage?**  
A: Freezing makes edge recovery a test of the patient-state representation learned during self-supervised pretraining, rather than letting the encoder over-specialize to the readout task.

**Q: What is DistMult doing here?**  
A: DistMult scores a source, relation, and target by the element-wise interaction of their latent vectors and a learned relation vector.

**Q: What is the target encoder?**  
A: It is a slowly updated copy of the context encoder. It receives no direct gradient; its weights track the online encoder through exponential moving average updates.

**Q: Why not reconstruct the original clinical note?**  
A: Reconstructing notes is expensive, privacy-sensitive, and may reward surface text generation. This model predicts latent graph structure, which is closer to the representation we want to improve.

**Q: What is the key novelty?**  
A: The main idea is to use self-supervised latent prediction over clinical knowledge graphs as a patient-state world model, then evaluate whether the frozen representation helps recover missing inferred relations.

**Q: How is this different from standard knowledge graph completion?**  
A: Standard KG completion usually operates over one large global graph. Fawkes operates over many patient-specific admission graphs and must infer relations from each local clinical context.

**Q: How is this different from a graph autoencoder?**  
A: The model predicts target latents through an EMA target branch rather than reconstructing raw adjacency or node content directly. The readout then tests whether the latent space supports relation recovery.

**Q: How is this different from using an LLM to answer questions over the note?**  
A: An LLM can reason over text, but Fawkes produces a structured graph representation with typed nodes and relations. That makes the output more auditable and easier to integrate with graph-based clinical workflows.

**Q: What are the important inferred relations?**  
A: The paper focuses on inferred clinical cross-links such as `MANAGED_FOR`, `CONFIRMS`, `COMPLICATED_BY`, and `INDICATES`.

## Data And Graph Construction

**Q: What is a patient-state knowledge graph?**  
A: It is a typed graph for one admission. Nodes represent entities like diagnoses, medications, procedures, services, microbiology, and the patient; edges represent relations among them.

**Q: What dataset do you use?**  
A: The available evaluation artifact is a 4,000-admission MIMIC-derived embedded JSONL with 186,334 nodes and 267,952 edges, including 768-dimensional note embeddings and provenance labels.

**Q: Are the data public?**  
A: The artifacts are MIMIC-IV-derived and remain subject to the PhysioNet data use agreement. The code and documentation can be shared, but raw JSONL, note-derived embeddings, and model artifacts require permission checks.

**Q: How are notes represented?**  
A: The paper checkpoint uses a 768-dimensional Clinical-ModernBERT note embedding.

**Q: What does "entity-localized note" mean?**  
A: The admission note embedding is copied only onto entities grounded by note-provenance evidence. Entities not grounded by the note receive a zero note vector.

**Q: Why localize one admission-level note vector instead of giving it to every node?**  
A: Localizing tells the model where narrative context is relevant. If every entity gets the same note vector, the model loses the signal about which entities the note actually supports.

**Q: How do you know which entities are note-grounded?**  
A: The released checkpoint uses provenance labels on edges. If an edge has `prov_in_note`, both endpoints are marked as note-grounded.

**Q: Does Fawkes use SapBERT entity embeddings?**  
A: No. The Fawkes paper implementation uses learned node-type embeddings, learned hashed-entity buckets, and a projected demographic/note branch.

**Q: How are entity names represented?**  
A: The normalized entity name is mapped through MD5 into one of 8,192 learned hash buckets.

**Q: Why use hashed entity buckets?**  
A: It is compact, deterministic, and trainable. The tradeoff is that it does not encode semantic similarity directly and can have collisions.

**Q: What are the demographic inputs?**  
A: The numeric branch starts with six values: normalized age, male indicator, female indicator, and three reserved zeros. In note mode, the 768-dimensional note vector is appended.

**Q: What is the correct input width for Fawkes note mode?**  
A: The numeric/note branch is 774 dimensions: six demographic values plus a 768-dimensional note embedding. It is projected to 128 dimensions and added to the type and hash embeddings.

**Q: Does the model build the original graph from raw MIMIC tables?**  
A: The repository consumes prebuilt graph JSONL records. It does not reproduce the upstream MIMIC extraction or LLM graph-generation pipeline end to end.

**Q: Are LLM-generated edges included?**  
A: Yes, the embedded dataset includes LLM-derived edges. In the released Fawkes checkpoint, no-evidence LLM-derived edges are pruned when they lack biomedical support and note provenance.

**Q: How many relations and node types are in the dataset?**  
A: The embedded data includes six node types and ten forward relations in the audited dataset; Fawkes internally recognizes a broader relation vocabulary with inverse types for message passing.

## Evaluation And Metrics

**Q: What is leave-one-out edge recovery?**  
A: Remove one true edge from the graph, re-encode the graph without it, then rank the true target against compatible candidate targets.

**Q: What candidate set is used?**  
A: In the Fawkes evaluator, candidates have the same node type as the true target, and other already-true targets for the same source and relation are filtered out.

**Q: Why use same-type negatives?**  
A: It prevents the task from being trivial. If the hidden target is a diagnosis, the model has to rank against other diagnosis candidates rather than invalid types.

**Q: What are Hits@1, Hits@3, and Hits@10?**  
A: They measure how often the true hidden target appears in the top 1, 3, or 10 ranked candidates.

**Q: Is MRR 0.419 clinically good?**  
A: It is evidence that the representation recovers hidden graph relations better than chance under the paper protocol. It is not, by itself, evidence of improved patient outcomes or deployment readiness.

**Q: What is the published result versus the full-file result?**  
A: The published number is MRR 0.418653 on the seeded 400-admission test split. The full-file evaluator result is MRR 0.440249 over 40,000 queries and should not be quoted as the published paper result.

**Q: Why are those two numbers different?**  
A: They evaluate different query populations. The published number uses the trainer's seeded test split and graph filter; the full-file result evaluates the whole 4,000-record file under the shipped evaluator.

**Q: What does the 400-admission test split mean?**  
A: The trainer uses a random seed of 42 and test fraction of 0.1, so 400 of the 4,000 admissions are used for the reported test-split evaluation.

**Q: How reproducible are the results?**  
A: Forward-only evaluation reproduces exactly. Training can drift slightly across CPU thread counts because floating-point reductions are not bit-stable above one thread.

**Q: Do the numbers compare directly to the `clinical_jepa` rows in the README?**  
A: No. Those rows use different architectures, data populations, and evaluation settings; the synthetic rows should not be interpreted as clinical performance comparisons.

**Q: What is cascade evaluation?**  
A: Cascade evaluation starts from deterministic backbone edges and adds inferred relation families in sequence to test how recovery changes as more inferred context becomes available.

**Q: What is the most likely evaluation criticism?**  
A: That edge recovery is an internal graph-completion metric, not a clinical endpoint. The honest answer is that this is representation-learning evidence, and clinical utility needs downstream validation.

**Q: What baseline would a reviewer ask for?**  
A: They may ask for same-type frequency baselines, classical KG completion, GNN without JEPA pretraining, note-to-all-nodes ablations, LLM rankers, and downstream task evaluations.

**Q: Do you compare against LLMs?**  
A: The repository includes benchmarking scaffolding for an LLM ranker, but the paper-facing result should be described in terms of the audited Fawkes leave-one-out protocol unless a like-for-like LLM result is available.

## Engineering And Implementation

**Q: What is the architecture?**  
A: Fawkes adds three 128-dimensional inputs per node: a learned type embedding, a learned hashed-entity embedding, and a projected demographics/note vector. Two four-head TransformerConv layers produce 128-dimensional node latents.

**Q: What library stack does it use?**  
A: The implementation is Python with PyTorch and PyTorch Geometric, with Clinical-ModernBERT embeddings supplied in the data artifact.

**Q: How large is the model?**  
A: The released Fawkes checkpoint is roughly 5.2 MB, with a 128-dimensional hidden width and two TransformerConv layers.

**Q: Is this computationally heavy?**  
A: The released configuration is relatively lightweight compared with large language models. Full evaluation over 4,000 records takes minutes on CPU in the documented setup.

**Q: Can it run without notes?**  
A: Yes. Fawkes has an Option A no-note setting with `USE_NOTE=0`, but the paper checkpoint being presented is the entity-note configuration.

**Q: What happens if a note embedding is missing?**  
A: For faithful evaluation of the note checkpoint, the 768-dimensional note embedding is required. Running with zero-filled note vectors is structurally possible but not a faithful test of the note-augmented model.

**Q: Are checkpoint configs validated?**  
A: Yes. The evaluator refuses to run if shape-affecting flags like note usage, grounding mode, embedding dimension, or score usage disagree with the checkpoint config.

**Q: Is the model deterministic?**  
A: Evaluation is deterministic. Training is seeded, but exact bitwise reproducibility requires controlling CPU threading.

**Q: What was the hardest engineering issue?**  
A: Preserving the paper implementation exactly while splitting a monolithic trainer into import-safe modules, then gating behavior against pre-restructure baselines.

**Q: Can this process new hospital data?**  
A: In principle, yes, if the hospital can produce the same kind of typed graph records, note embeddings, and provenance labels. The current repo does not include the full raw extraction pipeline.

**Q: How do you prevent invalid predictions?**  
A: The evaluation restricts candidates by target type; the broader modular pipeline also has schema-aware guards for graph revision. For deployment, stronger schema and clinical validation layers would be needed.

**Q: Can the model explain a prediction?**  
A: It can expose the graph context, relation type, candidate ranking, and provenance grounding. It is more inspectable than a raw black-box note classifier, but it is not a full causal explanation system.

**Q: What would productionization require?**  
A: Stable graph extraction, terminology normalization, provenance tracking, monitoring for drift, human review workflows, data-governance controls, and external validation.

## Researcher Questions

**Q: Why predict hidden node latents rather than hidden edges during pretraining?**  
A: Hidden-node prediction forces the encoder to model broader patient-state context before relation readout. Edge recovery is then used as a downstream probe of that latent space.

**Q: Why use TransformerConv instead of GINE or R-GCN?**  
A: TransformerConv gives relation-conditioned attention over local graph neighborhoods with a compact two-layer architecture. Other relational GNNs are reasonable ablations.

**Q: Does the model learn real medical semantics or dataset shortcuts?**  
A: It likely learns both graph regularities and clinical co-occurrence patterns. That is why external validation, temporal splits, and shortcut analyses are important future work.

**Q: How do you avoid information leakage from the note embedding?**  
A: The evaluation removes the target edge from message passing, but the note embedding may still contain narrative evidence about grounded entities. The claim is graph-relation recovery with note context, not note-blind inference.

**Q: Is the note embedding too coarse because one vector represents the whole admission?**  
A: That is a limitation. The localization strategy helps by assigning the vector only to grounded entities, but span-level or entity-specific note embeddings would be a natural improvement.

**Q: How do you evaluate calibration?**  
A: The reported paper metric is ranking-focused, not calibration-focused. Deployment would need calibrated confidence and threshold analysis.

**Q: Does the model handle temporal order?**  
A: The graph represents an admission-level state. Temporal ambiguity is one motivation for the work, but the released Fawkes model is not a full temporal trajectory model.

**Q: How does the model handle negation or absence?**  
A: The paper implementation focuses on typed graph relations in the supplied graph. Robust handling of negation, uncertainty, and absent findings depends on upstream extraction and downstream schema rules.

**Q: How would you test external validity?**  
A: Run on a separate institution or later MIMIC split with the same graph schema, report per-relation recovery, calibration, failure modes, and downstream utility.

**Q: What are the key ablations?**  
A: Note-free versus note-localized, note grounded by provenance versus name/all, JEPA pretraining versus readout-only, frozen versus unfrozen encoder, and different negative-sampling or candidate policies.

**Q: What is the chance baseline?**  
A: It depends on the candidate-set size per query. A careful comparison should report candidate counts and per-relation baselines, not a single global chance number.

**Q: Are the four inferred relations weighted?**  
A: Yes. `MANAGED_FOR`, `CONFIRMS`, `COMPLICATED_BY`, and `INDICATES` receive triple weight during readout training.

**Q: Why weight those relations?**  
A: They are the paper-focused inferred cross-links, so the training objective emphasizes recovery of the relations most relevant to graph revision.

**Q: Could the model simply memorize entity names?**  
A: Hashed entity embeddings can capture entity identity and frequency patterns, so memorization is a real concern. Leave-one-out tests local recovery; broader generalization needs held-out entities, hospitals, and time periods.

**Q: Why use MD5 buckets instead of ontology embeddings?**  
A: The hash bucket approach is simple and stable, but less semantically rich. Ontology-aware or language-model entity embeddings are plausible extensions.

**Q: Is the graph directed?**  
A: The input relations are directed, and Fawkes creates inverse relation types during message passing so direction-specific information can flow both ways.

**Q: Does the model produce new edges directly?**  
A: The paper evaluator ranks candidate targets for hidden edges. The broader modular pipeline exposes graph-revision actions, but Fawkes itself is best described as relation recovery and ranking.

## Investor Questions

**Q: What is the commercial opportunity?**  
A: The opportunity is better structured clinical intelligence: converting messy notes and EHR facts into cleaner patient-state graphs that can support review, search, risk workflows, trial matching, coding, and care coordination.

**Q: Is this a product or research?**  
A: It is research with a plausible product direction. The current artifact demonstrates representation learning and edge recovery; a product would need validated workflow integration.

**Q: Who would buy this?**  
A: Potential buyers could include health systems, clinical AI companies, life-sciences teams doing real-world evidence, and vendors that need structured patient context from EHR data.

**Q: What is the wedge use case?**  
A: A strong wedge is human-in-the-loop graph cleanup: flag missing or questionable relations in patient summaries before they feed downstream analytics or decision-support tools.

**Q: Why is this defensible?**  
A: Defensibility would come from high-quality graph extraction, clinical provenance, institution-specific validation data, integration into workflows, and accumulated feedback loops, not from the small model architecture alone.

**Q: Is there an FDA/regulatory issue?**  
A: If used only to organize information for human review, the regulatory burden may differ from autonomous clinical decision support. Any patient-care recommendation use would require careful regulatory strategy.

**Q: How far is this from deployment?**  
A: It needs external validation, prospective workflow evaluation, calibration, privacy/security review, and a production-grade extraction pipeline before clinical deployment.

**Q: What is the value over Epic/Cerner native data?**  
A: Native structured data misses many clinically meaningful relations buried in notes. This approach aims to connect structured facts and narrative evidence into an auditable graph.

**Q: What is the ROI story?**  
A: Possible ROI could come from reducing manual chart review, improving cohort identification, surfacing missing context, and improving downstream analytics. That still needs workflow-specific validation.

**Q: Is the model too small to be valuable?**  
A: Small can be an advantage in regulated environments: cheaper, easier to audit, and easier to run locally. The key is whether the representation improves the workflow metric.

**Q: Could a foundation model company copy this?**  
A: The architecture alone is copyable. The harder assets are graph construction, provenance, validation, deployment integrations, and clinical feedback data.

**Q: What moat would you build?**  
A: Proprietary evaluated graph pipelines, specialty-specific schemas, institution-level integrations, clinician review data, and measured outcome/workflow improvements.

**Q: What market do you start with?**  
A: Start where graph cleanup has immediate value and low autonomy risk: chart abstraction, patient summarization QA, registry/cohort building, or trial matching.

**Q: What is the biggest adoption blocker?**  
A: Trust. Clinicians and health systems will need clear provenance, failure-mode visibility, validation on their data, and workflow designs that save time rather than add another review queue.

## Business And Product Questions

**Q: Who is the user?**  
A: Near-term users are analysts, clinical informaticists, chart reviewers, and clinical AI teams. Later users could include clinicians if the system is embedded into reviewed workflows.

**Q: What is the product form factor?**  
A: A graph-revision layer or API that scores existing patient-graph edges, ranks missing relations, and exposes provenance for human review.

**Q: What would the UI show?**  
A: A patient graph, candidate missing relations, confidence/rank, source note evidence, and reasons for keeping, reviewing, pruning, or adding edges.

**Q: How do you keep humans in the loop?**  
A: Treat predictions as ranked suggestions, require review for clinical use, preserve provenance, and log reviewer decisions for monitoring and improvement.

**Q: What is the integration path?**  
A: Start offline on exported EHR/note data, then integrate with FHIR or vendor-specific interfaces once graph extraction and review workflows are validated.

**Q: What would a pilot measure?**  
A: Time saved in chart review, precision of suggested graph edits, recall of important missing relations, reviewer agreement, and downstream improvement in a concrete workflow.

**Q: Can it be specialty-specific?**  
A: Yes. Specialty-specific relation schemas and validation sets would likely be necessary for serious deployment.

**Q: Could this support trial matching?**  
A: Potentially, because trial matching often depends on relationships among diagnoses, medications, procedures, and evidence in notes. It would need task-specific validation.

**Q: Could this support coding or billing?**  
A: Potentially as a documentation/relationship-audit assistant, but billing use cases have strict compliance requirements and should be approached carefully.

**Q: Does this replace clinicians?**  
A: No. The intended direction is representation cleanup and review support, not autonomous clinical judgment.

## Clinical And Safety Questions

**Q: What clinical setting is this for?**  
A: The current data are hospital admissions from MIMIC-derived records. The method is more general, but claims should stay within this evaluated setting.

**Q: Is this validated prospectively?**  
A: No. The reported result is retrospective edge-recovery evaluation.

**Q: Does better edge recovery mean better care?**  
A: Not automatically. It suggests the graph representation is more complete under the evaluation protocol. Clinical impact would require downstream and prospective studies.

**Q: What are the failure modes?**  
A: Upstream extraction errors, note-provenance errors, shortcut learning, rare-entity collisions, missing temporal context, invalid candidate assumptions, and distribution shift.

**Q: What happens if the model suggests a wrong relation?**  
A: In a safe workflow, it should be a review suggestion with provenance, not an automatic change to clinical documentation or treatment.

**Q: How do you handle bias?**  
A: The current work does not establish fairness across subgroups. A deployment path would require subgroup evaluation and monitoring for graph-extraction and ranking disparities.

**Q: How do you handle privacy?**  
A: The data are governed by PhysioNet/MIMIC restrictions. A production system should run in a secure environment, minimize PHI exposure, and avoid unnecessary raw-note generation.

**Q: Is the model hallucinating?**  
A: It ranks schema-compatible candidate relations from a patient graph rather than free-generating text. It can still be wrong, but the error surface is more constrained and auditable.

**Q: Can clinicians inspect the evidence?**  
A: That is the intended safe design direction: show the graph context and note provenance behind the candidate relation.

**Q: Does the model know causality?**  
A: No. Relations like `MANAGED_FOR` or `COMPLICATED_BY` encode clinical associations in the graph, but the model is not proving causal mechanisms.

## Tough Or Skeptical Questions

**Q: Isn't this just learning from LLM-extracted labels, so it inherits LLM errors?**  
A: Yes, it can inherit extraction noise. The contribution is to model the graph as a noisy draft and recover plausible structure from context, but upstream extraction quality and human validation remain essential.

**Q: Why not just use GPT-5 on the note and ask for the relation?**  
A: For one-off reasoning, an LLM may be strong. Fawkes is aimed at structured, repeatable graph representation with typed candidates, provenance, and lower-cost local inference.

**Q: Are you evaluating on edges that came from the same note embedding given to the model?**  
A: The model uses note context, so some signal can come from narrative evidence. The fair claim is not note-blind prediction; it is whether localized note-aware patient-state latents recover hidden graph relations.

**Q: Could the model exploit degree or frequency patterns?**  
A: Yes, frequency shortcuts are possible in graph-completion tasks. Per-relation metrics, hard negatives, temporal/external splits, and entity-held-out tests would help quantify that.

**Q: Is MRR enough?**  
A: No. MRR is useful for ranking recovery, but product and clinical claims need precision at review thresholds, calibration, subgroup metrics, and downstream workflow outcomes.

**Q: Why is the headline result only on 400 test admissions?**  
A: The paper's reported protocol uses a seeded 10% test split from the 4,000-admission dataset. The full-file evaluator exists for reproducibility checks but is not the paper-reported number.

**Q: Are the synthetic v5/v6 numbers better than Fawkes?**  
A: They should not be compared. They use seeded synthetic graphs, different architectures, and different populations; they are regression gates, not clinical evidence.

**Q: Does this generalize outside MIMIC?**  
A: Not yet proven. External validation is one of the most important next steps.

**Q: Is the graph extraction pipeline itself validated?**  
A: The current repo focuses on the model, checkpoints, and evaluation over supplied graph artifacts. Full validation of upstream extraction is separate and necessary.

**Q: What is the biggest limitation?**  
A: The main limitation is that the result is retrospective relation recovery on MIMIC-derived graphs, not a prospective demonstration that the system improves clinical workflows or outcomes.

**Q: What would make the paper stronger?**  
A: Like-for-like baselines, external validation, stronger ablations around note localization, calibration analysis, and a downstream clinical workflow study.

## Quick Persona-Specific Prompts

**Q: Investor asks, "What is the business in plain English?"**  
A: Hospitals and healthcare AI teams need trustworthy structured patient context. This is a way to clean and complete patient graphs from notes and EHR facts so downstream tools work better.

**Q: Researcher asks, "What is the scientific claim?"**  
A: Self-supervised latent prediction over patient-specific clinical graphs can learn representations that support recovery of hidden clinically meaningful graph relations.

**Q: Engineer asks, "What do I need to run it?"**  
A: PyTorch, PyTorch Geometric, the released checkpoint, and graph JSONL records with typed nodes/edges plus note embeddings and provenance labels for the note model.

**Q: Clinician asks, "Would I trust this?"**  
A: Not as an autonomous decision-maker. I would trust it only as a reviewed suggestion layer with evidence, provenance, and local validation.

**Q: Business leader asks, "Where does it sit in the workflow?"**  
A: Between raw EHR/note extraction and downstream analytics or decision support: it scores and improves the patient graph before other systems consume it.

**Q: Reviewer asks, "What is your unfair advantage?"**  
A: The combination of patient-specific graphs, localized note evidence, self-supervised graph representation learning, and auditable relation recovery.

**Q: Skeptic asks, "What should I not conclude from this poster?"**  
A: Do not conclude that the model improves patient outcomes, is ready for autonomous clinical deployment, or has been externally validated.

**Q: Friendly attendee asks, "What is next?"**  
A: External validation, stronger baselines, richer note grounding, calibrated human-in-the-loop graph revision, and downstream workflow pilots.

