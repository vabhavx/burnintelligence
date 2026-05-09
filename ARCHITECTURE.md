# Architecture Deep Dive

## Pipeline Orchestration

The pipeline runs as a continuous loop with staggered intervals, orchestrated by `main.py`. Each stage has a hard timeout and health tracking.

```
Phase 1  [every 15 min]   GDELT GKG ingestion        (120s timeout)
         [every 6 min]    Bluesky Jetstream firehose   (360s timeout)
Phase 2  [every 10 min]   Sentence-transformer embed   (900s timeout)
Phase 3  [every 30 min]   Multi-resolution clustering  (1800s timeout)
Phase 4  [every 30 min]   Narrative DNA fingerprinting (7200s timeout)
Phase 5  [every 15 min]   NVI ensemble scoring         (900s timeout)
Phase 6  [every 15 min]   Lifecycle classification     (300s timeout)
Phase 7  [every 1 hour]   Cross-narrative analysis     (600s timeout)
Phase 8  [every 1 hour]   Graph topology analysis      (600s timeout)
Phase 9  [every 2 hours]  Source credibility scoring   (600s timeout)
Phase 10 [every 30 min]   Maintenance (retention, WAL) (300s timeout)
```

**Critical ordering invariant:** DNA (Phase 4) MUST run before NVI (Phase 5). The `cross_cluster_velocity` boost gate and `dna_match` cap gate both read live `dna_matches` table counts. If DNA hasn't run yet, `dna_match_count` is -1 (sentinel for "not computed"), and the DNA cap gate stamps NVI at 25.

### Startup Self-Test

On boot, `main.py` runs `selftest.py` which validates:
- All 18 gate functions are importable and callable
- The gate pipeline produces valid GateResult structs
- Terminal gate (gdelt_batch_artifact) correctly zeroes NVI
- Cap gates produce non-negative caps ≤100
- Boost gates produce floors in [0, 100]
- Force alert levels are "elevated" or "critical" when set

If any check fails, the engine refuses to start. This prevents deploying broken gate logic.

---

## Gate Pipeline Design

### Priority Order (Why It Matters)

Gates are ordered by specificity and destructiveness:

1. **gdelt_batch_artifact** (terminal) — If this fires, nothing else matters. The cluster is a GDELT ingestion artifact, not a real-world event. NVI = 0 immediately.

2. **content_noise** (suppress_only) — Listicles, obituaries, classifieds. Suppressed from the feed but NVI is preserved for internal tracking.

3. **insufficient_evidence** (cap 40/70) — Dynamic cap based on post count. <5 posts = 40 cap, 5-9 posts = 70 cap. Small samples are unreliable.

4. **single_source_cluster** (cap 15) — 1 domain or Shannon entropy <0.20. Cannot be cross-outlet coordination.

5. **wire_service** (cap 20/25) — Known syndicator domains or high hash diversity + high embedding similarity. Wire rewrites, not coordination.

6. **dna_match** (cap 25) — No DNA matches + small cluster (<20 posts). Real operators leave fingerprints across clusters.

7. **cross_language** (boost, floor 65) — 3+ real languages. Genuinely cross-cultural.

8. **geographic_spread** (boost, floor 60) — Locations spanning 3+ countries.

9. **high_signal_topic** (boost, floor 55) — Disinformation/propaganda themes with cross-source spread.

10. **circadian_anomaly** (boost, floor 50/65) — >40% posts during 1-5 AM UTC across diverse sources.

11. **content_anomaly** (boost, floor 50) — High activity density + high negative tone + low self-reference.

12. **cross_cluster_velocity** (boost, floor 65/80) — DNA evidence of same operator across multiple clusters. Two tiers: ≥5 high-confidence matches (elevated, floor 65) and ≥20 matches or cross-topic persistence (critical, floor 80).

13. **ensemble_uncertainty** (cap 35) — All 3 ensemble configs agree too perfectly on small samples.

14. **entity_concentration** (cap 35) — ≤1 shared named entity across ≥15 posts. Topic bags share taxonomy, not people.

15. **narrative_coherence** (cap 20) — Entity continuity + token Jaccard <0.22. Posts don't tell the same story.

16. **organic_viral_spread** (cap 30) — High mutation (>0.30) + high source diversity (>0.60). Independent rewrites by independent outlets.

17. **normal_news_cycle** (cap 35) — Low coordination (<1.05) + low burst (<1.5) + small cluster (<15 posts). Bypassed by source_diversity ≥0.7 or DNA matches ≥5.

18. **confidence_threshold** (suppress_only) — Confidence <0.65. Alert suppressed.

### Cap/Floor/Boost Architecture

```
Final NVI = max(min(raw_nvi, strictest_cap), highest_floor)

Where:
  strictest_cap = min(all cap values from capping gates that fired)
  highest_floor = max(all floor values from boosting gates that fired)
```

**Why boost beats cap:** A cluster with strong cross-cluster DNA evidence (12 domains, 113 high-confidence matches) should NOT be capped at 70 by insufficient_evidence just because it has 8 posts. The boost floor of 80 from cross_cluster_velocity correctly overrides the cap. This is the architectural decision that restores real detection.

### Why Falsification?

Positive detection ("prove this is coordinated") requires proving a negative — that the observed pattern could NOT have occurred organically. This is epistemologically impossible from open-source data alone.

Falsification inverts the burden: each gate tries to explain away the signal. If every gate fails to falsify it, the signal survives. This is Popperian: we can't prove coordination, but we CAN prove it's NOT a batch artifact, NOT a single source, NOT wire syndication, NOT a topic bag, NOT organic viral spread, etc.

What's left after falsification is the residue: clusters that COULD be coordinated and are worth human review.

---

## NVI Computation

### Raw NVI

```
raw_score = α·burst_zscore + β·spread_factor - γ·mutation_penalty + δ·tone_uniformity
nvi_raw = sigmoid(raw_score) × coordination_multiplier
```

**Burst (α·B):** Z-score of cluster post velocity vs. 7-day rolling baseline. Measures acceleration.

**Spread (β·S):** Composite of domain Shannon entropy, language count, and geographic diversity. Normalized to [0, 1].

**Mutation (γ·M):** Pairwise title token Jaccard distance. High mutation = posts are different (organic). Low mutation = posts are identical (syndication or copy-paste coordination). Penalized because high mutation makes coordination LESS likely.

**Tone Uniformity (δ·T):** Variance of GDELT tone scores across posts. Low variance = uniform emotional framing (suspicious). High variance = diverse perspectives (organic).

**Coordination Multiplier (C):** Temporal synchrony signal. Posts arriving in tight temporal clusters across diverse sources get a multiplier >1.0. Random arrival patterns get 1.0.

### Ensemble Approach

Three coefficient sets prevent single-signal evasion:

| Set | α (burst) | β (spread) | γ (mutation) | δ (tone) | Weight |
|-----|-----------|------------|--------------|----------|--------|
| A: Burst-heavy | 0.35 | 0.20 | 0.28 | 0.20 | 0.33 |
| B: Spread-heavy | 0.20 | 0.35 | 0.28 | 0.20 | 0.33 |
| C: Mutation-heavy | 0.25 | 0.25 | 0.30 | 0.20 | 0.34 |

If max(NVI) - min(NVI) > 20 across sets, the result is flagged "uncertain" (possible evasion). Final NVI = weighted mean of all three.

### Bias-Free Sigmoid

The sigmoid is centered at 0 with no baseline shift:
```
σ(x) = 100 / (1 + e^(-x/10))
σ(0) = 50
```

A neutral cluster (no burst, no spread, normal mutation, normal tone, coordination=1.0) scores exactly 50. This is the "baseline" — neither elevated nor suppressed. Real signal must earn its score through positive evidence.

---

## Narrative DNA Fingerprinting

### Design Goals

A coordinated operator can change domains, rewrite content, and shift posting schedules. But they CANNOT simultaneously control:
- Their writing style (stylometry)
- Their posting rhythm (cadence FFT)
- Their amplifier network topology
- Their entity relationships

DNA fingerprinting exploits this: it measures signals the operator CANNOT fake all at once.

### Dimension Details

**Stylometric (32-dim):**
- Function word frequency ratios (the, a, an, of, in, to, etc.)
- Sentence length distribution (mean, std, skew)
- Punctuation density
- Capitalization patterns
- Quote-to-text ratio

These are author-level signals. Two journalists from the same newsroom have different function word profiles. An operator writing under multiple bylines will have consistent stylometry.

**Cadence (16-dim):**
- FFT of posting timestamps (first 8 frequency magnitudes)
- Inter-arrival time statistics (mean, median, std, IQR)
- Hour-of-day distribution entropy
- Day-of-week distribution entropy
- Burstiness coefficient

Posting rhythm is hard to fake. A single operator running 5 personas will have consistent cadence across all of them because they sleep, eat, and work on a human schedule.

**Network (12-dim):**
- Amplifier graph: who amplifies whom
- Domain co-occurrence patterns
- Retweet/reshare topology
- Source diversity entropy
- Domain authority distribution

Network topology reveals infrastructure. Five personas sharing the same set of amplifier accounts form a recognizable subgraph.

**Entity Bias (24-dim):**
- Top-10 entity co-occurrence pairs
- Entity transition probabilities (Markov chain)
- Person-to-organization affinity
- Location-to-topic affinity
- Sentiment-toward-entity consistency

Who an operator writes about and how they feel about them is a persistent fingerprint. A propagandist who always pairs "NATO" with "aggression" and "Russia" with "defense" will do so across every persona and domain.

### Matching

Weighted cosine similarity with dimension weights [0.30, 0.30, 0.20, 0.20]. Threshold ≥0.75 = same operator.

High-confidence matches (≥0.90) require alignment across ALL four dimensions. Wire-syndicated content typically scores 0.75–0.85 because CMS-specific editor fingerprints differ in cadence/network dimensions even when content is identical. ≥0.90 means genuine operator persistence.

Cross-topic persistence is detected when dimension-level scores show:
- stylometric > 0.85 (same writer)
- cadence > 0.80 (same rhythm)
- entity_bias < 0.50 (DIFFERENT topics — same writer, different subject)

This is the strongest signal: the same author writing about unrelated stories for multiple outlets.

---

## Multi-Resolution Clustering

HDBSCAN is run at 4 density levels simultaneously:

| Resolution | min_cluster_size | What It Finds |
|-----------|-----------------|---------------|
| Fine | 3 | Small, tight clusters (breaking stories) |
| Medium | 5 | Medium clusters (developing narratives) |
| Coarse | 10 | Broad clusters (sustained coverage) |
| Macro | 25 | Very broad clusters (major events) |

After clustering, cross-resolution deduplication merges overlapping clusters using Jaccard similarity on member post IDs. A cluster found at resolution 3 that's 80% contained in a resolution-10 cluster is merged. This prevents the same narrative from appearing at multiple resolutions.

### Label Generation

Cluster labels are generated from actual article titles, not AI-generated summaries:
1. Extract all post titles in the cluster
2. Clean wire suffixes ("Title | Source Name" → "Title")
3. Find the most common cleaned title (dominant headline)
4. Fall back to GDELT entity metadata if no titles exist

This is an authenticity guarantee: the label you see is what the sources actually said.

---

## Database Schema

SQLite with WAL journaling for concurrent reads during writes.

Key tables:
- `posts` — Raw ingested articles with content_hash dedup
- `embeddings` — 384-dim sentence-transformer vectors
- `clusters` — Multi-resolution cluster assignments
- `nvi_snapshots` — NVI scores with full gate trace
- `dna_fingerprints` — 84-dim fingerprint vectors per cluster
- `dna_matches` — Cross-cluster fingerprint similarity scores
- `coordination_signals` — Persisted signal evidence
- `evidence_packs` — Generated evidence documents
- `pipeline_state` — Key-value state store
- `health` — Stage-level health tracking
- `source_scores` — Source credibility ratings
- `cross_narrative_links` — Inter-cluster relationship edges
- `lifecycle_states` — Cluster lifecycle classification

---

## Ingestion Details

### GDELT GKG

- Fetches `lastupdate.txt` every 15 minutes to find the latest GKG CSV
- Downloads and parses the zipped CSV (tab-separated, 20+ fields)
- Pre-filters for high-signal themes (disinformation, propaganda, cyber attacks, etc.)
- Extracts page titles from GKG extras, fixing mojibake encoding
- Stores translated and non-translated articles separately
- Content quality checks: printable text ratio, theme code detection, thin content rejection

### Bluesky Jetstream

- Connects to Jetstream WebSocket firehose
- Two-tier keyword matching: high-specificity terms OR 2+ low-specificity terms
- Minimum 30-character post length filter
- Language detection from post metadata
- 5-minute sessions, up to 500 posts per session

---

## API Design

FastAPI on port 8000. Key endpoints:

```
GET  /api/intelligence/narratives     — List all narrative clusters
GET  /api/intelligence/narrative/:id  — Single narrative with full detail
GET  /api/intelligence/evidence/:id   — Evidence pack with gate trace
GET  /api/intelligence/stats          — System statistics
GET  /api/intelligence/health         — Pipeline health status
GET  /api/health                      — Liveness check
```

The narratives endpoint supports filtering: `min_post_count`, `alert_level`, `language`, `region`, `category`, `source`. English feed includes "unknown" language articles since GDELT TLD-based language detection labels many English-language sites as "unknown."

Evidence packs include: full gate trace (which gates fired and why), NVI component breakdown, ensemble disagreement, DNA match evidence, anomaly signals, and falsification criteria. Everything is traceable to source data.

---

## Confidence Computation

Confidence is computed INDEPENDENTLY of NVI (unlike earlier versions that derived confidence from NVI magnitude). This prevents circular reasoning.

```
confidence = f(sample_size, source_diversity, signal_stability, gate_consistency)

sample_size:     logistic(post_count, midpoint=15, steepness=0.3)
source_diversity: direct Shannon entropy [0,1]
signal_stability: 1.0 - ensemble_disagreement/100
gate_consistency: 1.0 if <3 gates fired, decays with each additional gate
```

Final confidence is the product of these four independent factors. The confidence_threshold gate (gate 18) suppresses alerts below 0.65.

---

## Known Architectural Tradeoffs

1. **GDELT V1Locations use numeric IDs, not country codes.** The `geographic_spread` gate is disabled by default because GDELT's location field (e.g., "1#United States#US#...") uses an internal numbering scheme that doesn't reliably map to real countries.

2. **SQLite, not PostgreSQL.** Single-machine deployment. WAL mode enables concurrent reads. For multi-node deployments, this would need a migration to PostgreSQL.

3. **No real-time streaming.** The pipeline runs on intervals (15-min GDELT, 5-min Bluesky). True real-time would require a streaming architecture with Kafka or similar.

4. **Sentence transformers are compute-heavy.** Embedding 3M articles/day requires GPU acceleration for production. The current implementation targets CPU for portability.

5. **DNA is O(n²).** Cross-cluster fingerprint matching grows quadratically with cluster count. The 2-hour timeout and chunked processing prevent runaway, but at 10,000+ active clusters this becomes a bottleneck.

---

## Testing & Validation

- `selftest.py` — Gate pipeline integrity (runs on every startup)
- `synthetic_benchmark.py` — Synthetic data with known coordination patterns
- `evaluate.py` — Evaluation against ground truth labels
- `regression_baseline.json` — Known-good NVI scores for regression testing

Run tests: `make test`
Run benchmark: `make benchmark`
