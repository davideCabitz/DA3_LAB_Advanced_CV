# DA3 Exam Presentation: Recommended Structure

**Paper:** *Depth Anything 3: Recovering the Visual Space from Any Views*, Lin et al. (ByteDance Seed), arXiv 2511.10647, Nov 2025
**Format:** .ppt, 15 min + individual questions, groups of 2
**Grading:** presentation quality / understanding **50%**, novelty + technical feasibility of proposals **30%**, individual Q&A **20%**

> Built from a comparison of the 6 example decks in this folder (see Part 1). Part 2 is the recommended slide structure for DA3, and Part 3 covers Q&A preparation and a rules checklist.
> Technical details of DA3 marked **[verify]** come from my summary of the paper. Check them against the PDF before they go on a slide.

---

## Part 1: What the 6 example decks do

### 1.1 Overview

| Deck | Paper | Slides | Paper vs. own ideas (approx.) | Own experiments? | Structure in one line |
|---|---|---|---|---|---|
| `224824_221723_ACV_Presentation.pdf` | KD for mobile monocular depth (Wang 2021) | 27 | Background 6 / Paper 9 / **Ours 9 (~35%)** | No (architecture proposal + expected numbers from MobileViT paper) | TOC → Background → Paper → "Our topics" with section dividers |
| `ACV-ViT.pdf` | ViT (Dosovitskiy 2021) | 22 | Background 5 / Method 6 / Exp 4 / **Ours 1 (~5%)** | No | Classic textbook walk-through, improvements as a final bullet list |
| `ACV_SAGNN.pdf` | Scale-Aware GNN for few-shot segmentation | 22 | Paper 12 / **Ours 7 (~32%)** | Tiny (learned similarity tried on PFENet) | Outline → pipeline built up progressively → results → 6 concrete modifications |
| `ACV_presentation.pdf` | AdvProp (Xie 2020) | 32 | Paper 14 / **Ours 13 (~40%)** | **Yes** (EfficientNet-B0 on SVHN, 3 variants + robustness test) | Intro → Basics → Related → Method → *critique* → hypothesis → experiments → conclusions → **appendix** |
| `GIRAFFE.pdf` | GIRAFFE (Niemeyer 2021) | 32 | Lineage 11 / Paper 17 / **Ours 3 (~10%)** | No | Story: GAN → NeRF → GRAF → GIRAFFE, very visual, one open-question proposal |
| `InterFaceGAN.pdf` | InterFaceGAN (Shen 2020) | 29 | Paper 22 / **Ours 3 (~10%)** | No | Outline → assumptions → step-by-step → demos → a found limitation → "minor/major improvement" |

### 1.2 What the strong decks do well (adopt these)

1. **A critical analysis drives the proposals** (AdvProp, InterFaceGAN). AdvProp states *"the paper does not address robustness"*, makes a hypothesis, and tests it. InterFaceGAN shows a failure case (inversion overfits) and then proposes a fix. Proposals derived from a limitation sound much more convincing than a list of "things we could try".
2. **Real proof-of-concept numbers** (AdvProp, and partly SAGNN). AdvProp is the only deck with its own experiments, and the result is the strongest improvements section. The rules make this optional, but it is the clearest way to earn the 30% for feasibility.
3. **Building the main figure progressively** (SAGNN slides 4–7). The same pipeline figure appears on several slides, each one highlighting the next block (backbone → node embedding → GNN). The audience never gets lost.
4. **A lineage-style story** (GIRAFFE). Each predecessor (GAN → NeRF → GRAF) gets one slide ending in "what is missing", so the paper arrives as the answer. For DA3 the lineage is: SfM/MVS → DUSt3R → VGGT → DA1/DA2 → DA3.
5. **Navigation aids** (AdvProp breadcrumb bar *Intro · Basics · Related · AdvProp · Improvements* on every slide; KD's coloured section dividers and section chips like "ARCHITECTURE" / "FRAMEWORK"). With 15 minutes and two speakers, a breadcrumb costs nothing and helps a lot.
6. **One highlighted takeaway per slide** (InterFaceGAN: a red sentence or green check, e.g. *"The latent space is semantically linear separable ✓"*). The examiner should be able to read the conclusion of a slide in 2 seconds.
7. **Redrawn diagrams** (KD deck: teacher/student, KD loss, MobileViT block all redrawn in the deck's own style). This shows understanding, and it complies with the "no slide reuse from the web" rule.
8. **An appendix for Q&A** (AdvProp: EfficientNet slide after the references). Backup slides let you answer the 20% questions with a figure instead of from memory.

### 1.3 Weaknesses to avoid

| Weakness | Where | Why it hurts with this rubric |
|---|---|---|
| Improvements = **existing published works** ("SOTA improvements: DeiT, Swin"; "major improvement: GANSpace") | ViT, InterFaceGAN | Zero novelty. The 30% is for *your* ideas. Cite prior work only to position your idea against it. |
| Improvements as a **bullet list with no "why/how"** | ViT, KD ("Implementation on TinyML": 3 bullets) | No technical feasibility shown: no data, no metric, no cost. |
| **Too much generic background** (TinyML advantages, "real-life examples", what a Transformer is) | KD, ViT | It takes time away from the paper and the proposals, and the examiner already knows it. |
| **32 slides for 15 min** | AdvProp, GIRAFFE | About 28 s per slide means rushing. Aim for about 18–20 main slides (≈45 s each) and put the rest in the appendix. |
| **Full results tables with nothing highlighted** | SAGNN (COCO/PASCAL tables), ViT | Nobody can read them in 30 s. Crop to the relevant rows and colour your point. |
| **No outline, no references slide** | GIRAFFE | Looks unfinished. References are also needed to credit borrowed figures. |
| **6 slides of qualitative images** (controllable generation) | GIRAFFE | Pretty, but low on "understanding". Two are enough. |
| Ideas listed but **not ranked or compared** | SAGNN (6 ideas, equal weight) | It reads as a brainstorm. Rank them and go deep on one. |

### 1.4 Common skeleton (what all 6 share)

`Title → Outline → Problem/Motivation → Background/Related → Method → Results → (Critique) → Proposals → Conclusion → References → (Appendix)`

The best decks differ from the weaker ones in three ways:
- **(a)** how much of the talk goes to *own* ideas (≥35%),
- **(b)** whether those ideas come from an identified limitation,
- **(c)** whether there is any experimental evidence.

---

## Part 2: Recommended structure for the DA3 talk

### 2.1 Design principles (derived from Part 1)

- **Time split** that mirrors the rubric: about **55% paper, about 10% critique, about 35% proposals**.
- **About 20 main slides + backup appendix.** One message per slide, with that message written as the slide title (e.g. *"One plain transformer is enough"*, not "Architecture").
- **Breadcrumb bar** on every slide: `Problem · Method · Results · Limits · Proposals`.
- **Redraw** the depth-ray and architecture diagrams yourselves. You can use paper figures and result images with a caption *"Source: Lin et al., 2025"*. **Never** copy slides found online.
- Keep **one recurring visual**: the architecture figure, built up step by step (SAGNN trick), shown again in the proposals with your modification highlighted in a different colour.
- Every proposal slide answers **What? Why (which limitation)? How? How do we measure it? What does it cost?**

### 2.2 Time budget (15:00)

| Block | Slides | Time | Speaker (suggested) |
|---|---|---|---|
| A. Opening & problem | 0–3 | 2:00 | A |
| B. Method | 4–8 | 4:30 | A |
| C. Results | 9–12 | 2:30 | B |
| D. Critical analysis | 13 | 1:00 | B |
| E. Proposals | 14–19 | 4:30 | B (P1) → A (P2, P3, feasibility) |
| F. Closing | 20 | 0:30 | B |

> Each person presents part of the paper **and** part of the proposals, so both look competent on both halves. The Q&A is individual, so **both must be able to explain every slide**.

### 2.3 Slide-by-slide plan

#### A. Opening & problem (~2:00)

**0. Title**
DA3 title, authors, venue/arXiv, course + A.Y., your names. Put one striking visual on it: a few casual photos → a 3D point cloud with camera frustums. Make it yourselves by running the released code.

**1. Outline**
The 5 breadcrumb sections. Mention up front: *"In the second half we propose X and show a proof of concept"*, so the examiner knows the improvements are coming.

**2. The problem: recovering the visual space from *any* views**
- Input: 1 image, N unordered images, or a video, with or without known camera poses.
- Output: per-view depth + camera pose, giving a consistent 3D scene.
- Why it matters: SLAM, AR, robotics, novel-view synthesis, autonomous driving.
- Takeaway box: *"Today these tasks need separate specialised models. Can one model do all of them?"*

**3. How we got here (lineage, GIRAFFE-style)**
A timeline with one "what's missing" note per step:
- Classical SfM/MVS (COLMAP): accurate but slow, brittle, needs many views.
- DUSt3R / MASt3R: feed-forward pairwise point maps, but pairs need global alignment.
- VGGT: one large network for many views, but it is a multi-task design with several heads/targets and a specialised architecture.
- Depth Anything 1/2: strong monocular depth, but single view only and no camera.
- **DA3's question:** *"What is the minimal set of modelling choices needed?"*

#### B. Method (~4:30)

**4. Two key insights** (the slide the examiner must remember)
1. **A single plain transformer** (vanilla DINOv2 encoder) is enough. No architectural specialisation is needed.
2. **A single depth-ray prediction target** is enough. There is no need for multi-task learning with separate point-map / pose / depth objectives.
Plus the *teacher-student* training paradigm, and training **only on public academic datasets**.

**5. The depth-ray representation** (redraw this yourselves)
- Per pixel: depth **D** + ray **r = (origin t, direction d)**, so the 3D point is **P = t + D · d**.
- Why it's better than a point map: depth and camera are disentangled, each target is "local" to its view, and consistency across views comes almost for free. **[verify the paper's exact argument + ablation]**
- How the camera pose is recovered from the ray map (fit rotation/translation/intrinsics to the predicted rays), plus the optional lightweight camera head. **[verify]**
- Put the derivation in the backup slides (B1). It's a likely question.

**6. Architecture** (progressive build: 2–3 animation steps or slides)
- Step 1: patch tokens from each view → **DINOv2** backbone.
- Step 2: **input-adaptive cross-view attention**: some layers attend within a view, others across all views, so the same weights handle 1 or N views. **[verify the layer split]**
- Step 3: optional **camera token** (pose conditioning when poses are known, a learned token otherwise) → **Dual-DPT head** with shared features and two outputs (depth, ray).

**7. Training: teacher-student**
- Problem: real datasets have noisy or sparse depth.
- A teacher (a monocular model trained on large synthetic data) produces dense pseudo-depth for the real data, aligned to the sparse ground truth (scale/shift fitting). **[verify details]**
- Losses: depth (+ confidence), ray, point/consistency, gradient terms. **[verify list]** Put the formulas in backup slide B3.

**8. One model, many uses** (keep it short)
- Model family (Giant/Large/Base/Small) and variants (monocular, metric, nested). **[verify names]**
- A feed-forward 3D Gaussian Splatting head for novel-view synthesis.
- DA3-Streaming for long videos (chunked, < 12 GB GPU).

#### C. Results (~2:30)

**9. A new visual geometry benchmark**
Tasks (camera pose, any-view geometry, rendering), datasets, metrics. Show one compact table: task → metric → what it measures.

**10. Main quantitative results**
- One cropped table with DA3 vs VGGT and the strongest baselines highlighted.
- Headline: **+44.3% camera pose accuracy, +25.1% geometric accuracy vs VGGT**, and better than DA2 on monocular depth.

**11. Qualitative results**
Two panels at most: a reconstruction compared with VGGT, and a novel-view-synthesis or video example. If possible, include one result you produced yourselves with the released code. It proves you ran it, and it prepares the proof of concept.

**12. Ablations: why the minimal design works**
The 2–3 ablations that support the key insights: depth+ray vs. other target combinations, single vs. specialised backbone, effect of the teacher. **[verify which ablations the paper reports]** This is the "understanding" slide, so explain *why*, not just numbers.

#### D. Critical analysis (~1:00)

**13. Strengths & limitations** (the bridge to the proposals, AdvProp-style)
- ✅ Simple, general, scales well, public data only, code released.
- ❌ **Static-scene assumption**: training and evaluation are on static scenes, so moving objects break cross-view consistency and corrupt the pose estimate.
- ❌ **Pure geometry**: no semantics, language, or interaction.
- ❌ Depth can be **over-smoothed** on fine details (reported by DepthWorld, CoRL 2026, for manipulation).
- Quote the paper's own future work: *"extend its reasoning to dynamic scenes, integrate language and interaction cues, and explore larger-scale pretraining…"*
- Takeaway: *"We tackle the first two. The third needs industrial compute."*

#### E. Proposals (~4:30) — the 30%

**14. What has already been done since DA3 (positioning, which proves novelty)**
A gap table like this one shows the examiner you know the field and that your idea is new:

| Direction | Done on DA3? | Done on similar models (VGGT)? | Our proposal |
|---|---|---|---|
| Dynamic scenes, training-free | Dyna3 (Oct 2026): VLM + SAM 3 + feature matching | VGGT4D | **P1**: geometry-only motion masking, no extra models |
| Dynamic scenes, fine-tuning | ✗ not found | PAGE-4D, V-DPM, DynamicVGGT | **P2**: depth-ray-**flow** with LoRA |
| Language | Thinking with Spatial Code, SpatialSV (DA3 as encoder or teacher) | Spatial-MLLM, VG-LLM | **P3** (outlook) |
| Interaction / world models | PAIWorld (3D-REPA from frozen 3D models) | — | P3 (outlook) |

**15. P1: Motion-aware DA3 (training-free), main proposal**
- *Why:* moving pixels violate the static assumption and bias the pose recovered from the ray maps.
- *What:* use DA3's own outputs to find motion. Reproject frame *i*'s 3D points (from P = t + D·d) into frame *j* and compare with *j*'s predicted depth/rays. A large residual or low confidence marks a dynamic pixel. Then re-solve the camera pose **using only the static pixels** (robust or weighted fitting) and optionally re-run with dynamic regions masked out of cross-view attention.
- *Difference from Dyna3:* no VLM, no SAM, nothing to train. It uses only geometric self-consistency, so it is cheaper and simpler.
- Figure: the architecture from slide 6 with the new "consistency check → mask → pose re-solve" block in a highlight colour.

**16. P1: Proof of concept** (optional per the rules, but strongly recommended: AdvProp's deck had the strongest proposals section of the six because of its experiments)
- Setup: released DA3 code, a small model (Base/Small) on 1 GPU or Colab.
- Data: a dynamic sequence benchmark, e.g. Sintel, Bonn RGB-D Dynamic, or TUM-RGBD dynamic sequences.
- Metrics: camera ATE / RPE, depth AbsRel / δ1 on dynamic vs. static pixels.
- Table: DA3 vanilla vs. DA3 + P1 (vs. Dyna3's numbers if they are comparable).
- If you can't finish: show the *protocol* + a qualitative mask figure + the expected outcome. That still scores on feasibility.

**17. P2: DA3-4D: depth-ray-flow** (training-based, medium cost)
- *What:* add a third DPT branch that predicts per-pixel **scene flow**, so the "single target" idea extends to 4D (depth + ray + flow).
- *How:* freeze most of DA3 and fine-tune only the middle layers with LoRA (PAGE-4D showed this is enough for VGGT) on synthetic dynamic data (e.g. PointOdyssey, Dynamic Replica, Kubric).
- *Why novel:* this has been done for VGGT but not for DA3, and it keeps DA3's minimal philosophy.
- *Risk:* domain gap from synthetic data, and the GPU budget (keep it to the Small/Base models).

**18. P3: Outlook: language and interaction**
- DA3 as the geometry encoder of a spatial MLLM: swap VGGT for DA3 in an open model (e.g. Spatial-MLLM) and evaluate on part of VSI-Bench.
- DA3 as a frozen 3D teacher for world models or VLA policies (REPA-style feature alignment).
- Present this honestly as a direction, not a contribution, because parts of it already exist (slide 14).

**19. Feasibility comparison** (shows judgement; SAGNN's six unranked ideas lacked this)

| | P1 Motion-aware | P2 Depth-ray-flow | P3 DA3 + LLM |
|---|---|---|---|
| Training needed | None | LoRA fine-tune | Fine-tune the MLLM |
| Compute | 1 GPU, hours | 1–4 GPUs, days | Multi-GPU, days–weeks |
| Data | Public dynamic benchmarks | Synthetic 4D datasets | VSI-Bench / ScanNet QA |
| Novelty vs prior work | Medium–high (geometry-only vs Dyna3) | High (none for DA3) | Low–medium |
| Risk | Low | Medium | High |
| Status | **PoC done / in progress** | Designed | Outlook |

#### F. Closing (~0:30)

**20. Conclusions** (3 lines)
1. DA3 shows that a plain transformer and a single depth-ray target are enough for any-view geometry.
2. Its main limitation is the static-world assumption.
3. We proposed a training-free fix (with PoC) and a 4D extension that keeps DA3's minimal design.

**21. References** (the DA3 paper, DA1/DA2, VGGT, DUSt3R, DINOv2, Dyna3, PAGE-4D, V-DPM, Spatial-MLLM, Thinking with Spatial Code, PAIWorld, the datasets used)

**Thank you / Q&A** (can share the slide with the references)

#### Appendix: backup slides (not presented, used during Q&A)

- **B1:** Recovering camera pose from the ray map (derivation).
- **B2:** Cross-view attention details (layer split, token rearrangement, cost vs. number of views).
- **B3:** Loss functions with formulas.
- **B4:** Extra tables: monocular depth vs DA2, the full pose table, NVS results.
- **B5:** Comparison table of DUSt3R vs VGGT vs DA3 (inputs, targets, heads, training data, size).
- **B6:** P1 implementation details (thresholds, failure cases).
- **B7:** P2 architecture + training recipe.

---

## Part 3: Q&A preparation (20%) and rules checklist

### 3.1 Questions to rehearse (each person answers all of them)

1. Why a depth + ray target instead of a point map, or of predicting the pose directly?
2. How exactly is the camera pose recovered from the ray map? What happens with noisy rays?
3. How does one network handle 1 view and N views with the same weights?
4. Why teacher-student? What is the teacher trained on, and why not just use the ground truth?
5. What exactly is the difference from VGGT? Why does DA3 win on pose by so much?
6. Is the output metric or up-to-scale? How does the metric variant work?
7. What breaks with moving objects, and why? (This connects directly to P1.)
8. Why DINOv2 and not another backbone? What does pretraining bring?
9. P1: what if most of the frame is dynamic? How do you choose the threshold? How does it compare with Dyna3?
10. P2: why would LoRA on the middle layers be enough? Where would the scene-flow supervision come from?
11. Complexity: how does the cost grow with the number of views? (Global attention is quadratic in tokens.)

### 3.2 Rules checklist

- [ ] DA3 is on the official paper list, or has been **validated by the instructors** if it isn't.
- [ ] Delivered as a **.ppt/.pptx** file.
- [ ] Rehearsed to **≤15:00** with both speakers (time each block against 2.2).
- [ ] **No slides reused from the web.** Diagrams redrawn, paper figures captioned *"Source: Lin et al., 2025"*.
- [ ] The talk includes **summary + improvement ideas** (both required). A PoC is optional but recommended.
- [ ] Every proposal states: limitation → idea → method → metric → cost.
- [ ] Slide numbers, breadcrumb bar, references slide, appendix.
- [ ] Both members can answer every question in 3.1.
