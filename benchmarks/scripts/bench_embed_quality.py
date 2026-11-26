#!/usr/bin/env python3
"""
Embedding quality benchmark for highperfasr.

Evaluates the QUALITY of embeddings produced by /v1/embeddings using
standard academic metrics (MTEB-aligned): nDCG@10 for retrieval,
Spearman for STS, V-measure for clustering, accuracy for classification.

Also measures Matryoshka dimension reduction curves (768/512/256/128).

Companion to bench_embed.py which measures SERVING performance (throughput/latency).

Usage:
    python3 bench_embed_quality.py --server http://localhost:8000
    python3 bench_embed_quality.py --server http://localhost:8000 --suite retrieval
    python3 bench_embed_quality.py --server http://localhost:8000 --suite all --dimensions 768,512,256,128
    python3 bench_embed_quality.py --server http://localhost:8000 --suite quick
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent.parent))

from preflight import detect_server, ensure_unbuffered, log_preflight_summary

ensure_unbuffered()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bench_embed_quality")

# --- Standard evaluation datasets (curated MTEB subset) ---

RETRIEVAL_DATASETS = {
    "science_qa": {
        "description": "Science Q&A retrieval — 30 queries against 30 passages",
        "metric": "nDCG@10",
        "queries": [
            "What causes tides on Earth?",
            "How do vaccines work?",
            "What is photosynthesis?",
            "How does DNA replication occur?",
            "What is the theory of relativity?",
            "How do antibiotics fight bacteria?",
            "What causes earthquakes?",
            "How does the human immune system work?",
            "What is quantum entanglement?",
            "How do solar panels generate electricity?",
            "What is the greenhouse effect?",
            "How does natural selection drive evolution?",
            "What causes lightning?",
            "How do neurons transmit signals?",
            "What is nuclear fusion?",
            "How does the water cycle work?",
            "What are black holes?",
            "How do enzymes catalyze reactions?",
            "What is plate tectonics?",
            "How does CRISPR gene editing work?",
            "What is dark matter?",
            "How do magnets work?",
            "What causes the seasons?",
            "How does the heart pump blood?",
            "What is the Doppler effect?",
            "How does machine learning work?",
            "What is entropy in thermodynamics?",
            "How do plants absorb water?",
            "What are gravitational waves?",
            "How does mRNA carry genetic information?",
        ],
        "corpus": [
            "Tides are caused primarily by the gravitational pull of the Moon on Earth's oceans. The Sun also contributes, but its effect is about half that of the Moon due to distance. Spring tides occur when the Sun and Moon align.",
            "Vaccines work by introducing a harmless form of a pathogen to stimulate the immune system. This creates memory cells that recognize and fight the real pathogen if encountered later, providing immunity without causing disease.",
            "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It occurs in chloroplasts using chlorophyll pigments that absorb light energy.",
            "DNA replication is a semi-conservative process where each strand serves as a template. Helicase unwinds the double helix, and DNA polymerase synthesizes new complementary strands in the 5' to 3' direction.",
            "Einstein's theory of relativity includes special relativity (E=mc², time dilation at high speeds) and general relativity (gravity as curvature of spacetime caused by mass and energy).",
            "Antibiotics fight bacteria through several mechanisms: disrupting cell wall synthesis (penicillin), inhibiting protein synthesis (tetracycline), or interfering with DNA replication (fluoroquinolones). They do not work against viruses.",
            "Earthquakes are caused by sudden release of energy in Earth's crust, typically along fault lines where tectonic plates meet. The energy radiates as seismic waves from the focus point.",
            "The human immune system has innate defenses (skin, mucus, white blood cells) and adaptive immunity (T-cells, B-cells, antibodies). It identifies pathogens through antigens and develops specific responses.",
            "Quantum entanglement is a phenomenon where two particles become correlated so that measuring one instantly affects the other regardless of distance. Einstein called it spooky action at a distance.",
            "Solar panels use photovoltaic cells made of semiconductor materials like silicon. When photons hit the cells, they knock electrons loose, creating an electric current through the photovoltaic effect.",
            "The greenhouse effect occurs when certain gases (CO2, methane, water vapor) in the atmosphere trap infrared radiation from Earth's surface, warming the planet. Without it, Earth would be about 33°C colder.",
            "Natural selection occurs when individuals with traits better suited to their environment survive and reproduce more successfully. Over generations, beneficial traits become more common in the population.",
            "Lightning is caused by electrical charge separation within clouds. Ice crystals and water droplets collide, building positive charges at the top and negative at the bottom, eventually discharging as lightning.",
            "Neurons transmit signals through electrical impulses (action potentials) along axons. At synapses, neurotransmitters are released across the gap to activate or inhibit the next neuron.",
            "Nuclear fusion is the process of combining light atomic nuclei to form heavier ones, releasing enormous energy. It powers the Sun, where hydrogen nuclei fuse into helium at extreme temperatures and pressures.",
            "The water cycle involves evaporation from surfaces, transpiration from plants, condensation into clouds, and precipitation as rain or snow. Water flows through rivers to oceans, completing the cycle.",
            "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse. The boundary is called the event horizon.",
            "Enzymes are biological catalysts that speed up chemical reactions by lowering activation energy. They bind to substrates at their active site, forming an enzyme-substrate complex with high specificity.",
            "Plate tectonics is the theory that Earth's outer shell is divided into plates that float on the mantle. Their movement causes earthquakes, volcanic activity, mountain building, and ocean trench formation.",
            "CRISPR-Cas9 is a gene editing tool that uses a guide RNA to direct the Cas9 enzyme to a specific DNA location, where it makes a precise cut. The cell's repair mechanisms then modify the gene.",
            "Dark matter is an invisible form of matter that doesn't emit or absorb light. It makes up about 27% of the universe and is detected through its gravitational effects on visible matter and galaxy rotation curves.",
            "Magnets work because of the alignment of magnetic domains — groups of atoms whose electrons spin in the same direction. This creates a magnetic field with north and south poles.",
            "Seasons are caused by Earth's 23.5-degree axial tilt. As Earth orbits the Sun, different hemispheres receive more direct sunlight at different times, creating summer and winter.",
            "The heart pumps blood through a double circulatory system. The right side sends deoxygenated blood to the lungs; the left side pumps oxygenated blood to the body through arteries.",
            "The Doppler effect is the change in frequency of a wave relative to an observer moving relative to the source. Sound pitch increases as a source approaches and decreases as it recedes.",
            "Machine learning is a subset of AI where algorithms learn patterns from data without explicit programming. Types include supervised learning (labeled data), unsupervised (clustering), and reinforcement learning.",
            "In thermodynamics, entropy is a measure of disorder or randomness in a system. The second law states that entropy in an isolated system always increases, driving processes toward equilibrium.",
            "Plants absorb water through root hairs by osmosis, moving from high to low water concentration. Water travels upward through xylem vessels via capillary action and transpiration pull.",
            "Gravitational waves are ripples in spacetime caused by accelerating massive objects. First detected by LIGO in 2015 from merging black holes, confirming Einstein's general relativity prediction.",
            "Messenger RNA (mRNA) is transcribed from DNA in the nucleus and carries genetic code to ribosomes in the cytoplasm, where it is translated into protein sequences using codons.",
        ],
    },
    "tech_faq": {
        "description": "Technical FAQ retrieval — 20 queries against 20 answers",
        "metric": "nDCG@10",
        "queries": [
            "How to fix a segmentation fault in C?",
            "What is a deadlock in concurrent programming?",
            "How does garbage collection work in Java?",
            "What is the difference between TCP and UDP?",
            "How to optimize SQL query performance?",
            "What is a race condition?",
            "How does HTTPS encryption work?",
            "What is containerization in DevOps?",
            "How to handle memory leaks in Python?",
            "What is the CAP theorem in distributed systems?",
            "How does a hash table work?",
            "What is the difference between REST and GraphQL?",
            "How to implement caching strategies?",
            "What is event-driven architecture?",
            "How does virtual memory work?",
            "What is dependency injection?",
            "How to prevent SQL injection attacks?",
            "What are microservices?",
            "How does a load balancer work?",
            "What is continuous integration and continuous deployment?",
        ],
        "corpus": [
            "A segmentation fault occurs when a program accesses memory it shouldn't. Debug with valgrind or address sanitizer. Common causes: null pointer dereference, buffer overflow, use-after-free, stack overflow from deep recursion.",
            "A deadlock occurs when two or more threads wait for resources held by each other, creating a circular dependency. Prevention strategies include lock ordering, timeout-based acquisition, and deadlock detection algorithms.",
            "Java's garbage collector automatically reclaims memory from unreferenced objects. It uses generational collection: young generation (Eden, Survivor spaces) with minor GC, and old generation with major GC. G1 is the default collector.",
            "TCP provides reliable, ordered, error-checked delivery using connections and acknowledgments. UDP is connectionless with no delivery guarantee but lower latency. TCP suits web/email; UDP suits streaming/gaming.",
            "Optimize SQL queries by: adding proper indexes, avoiding SELECT *, using EXPLAIN to analyze plans, reducing joins, partitioning large tables, using connection pooling, and avoiding N+1 query patterns in ORMs.",
            "A race condition occurs when multiple threads access shared data concurrently and the outcome depends on execution timing. Fix with mutexes, semaphores, atomic operations, or lock-free data structures.",
            "HTTPS uses TLS to encrypt HTTP traffic. The handshake involves: client hello, server certificate exchange, key agreement (using asymmetric encryption), then symmetric encryption for data transfer using the shared secret.",
            "Containerization packages applications with their dependencies into isolated containers. Docker creates lightweight images; Kubernetes orchestrates container deployment, scaling, and management across clusters.",
            "Python memory leaks often come from circular references, global caches, or C extensions. Use tracemalloc to track allocations, weakref for caches, gc module for circular reference detection, and memory profilers like pympler.",
            "The CAP theorem states distributed systems can guarantee at most two of: Consistency (all nodes see same data), Availability (every request gets a response), and Partition tolerance (system works despite network splits).",
            "A hash table stores key-value pairs using a hash function to compute an index into an array of buckets. Collisions are handled by chaining (linked lists) or open addressing (probing). Average O(1) lookup time.",
            "REST uses fixed endpoints with HTTP verbs and returns full resources. GraphQL has a single endpoint where clients specify exactly what data they need using queries, reducing over-fetching and under-fetching.",
            "Caching strategies include: write-through (write to cache and DB), write-back (write to cache, async to DB), cache-aside (app manages cache), TTL-based expiration, and LRU/LFU eviction policies.",
            "Event-driven architecture uses events to trigger communication between decoupled services. Components produce and consume events through message brokers (Kafka, RabbitMQ). It enables loose coupling and scalability.",
            "Virtual memory maps virtual addresses to physical RAM using page tables managed by the OS. When RAM is full, pages are swapped to disk. The TLB caches recent translations for performance.",
            "Dependency injection is a design pattern where objects receive their dependencies from external sources rather than creating them. It promotes loose coupling, testability, and follows the inversion of control principle.",
            "Prevent SQL injection by: using parameterized queries/prepared statements, input validation, escaping special characters, using ORMs, applying least-privilege database permissions, and using WAFs.",
            "Microservices architecture decomposes applications into small, independent services that communicate via APIs. Each service owns its data and can be deployed, scaled, and updated independently.",
            "A load balancer distributes incoming traffic across multiple servers using algorithms like round-robin, least connections, or IP hash. It improves availability, reliability, and performance through horizontal scaling.",
            "CI/CD automates building, testing, and deploying code. CI merges code frequently with automated tests. CD automatically deploys tested code to staging/production. Tools include Jenkins, GitHub Actions, and GitLab CI.",
        ],
    },
}

STS_DATASETS = {
    "stsb": {
        "description": "STS Benchmark — semantic textual similarity",
        "pairs": [
            ("A plane is taking off.", "An air plane is taking off.", 5.0),
            ("A man is playing a large flute.", "A man is playing a flute.", 3.8),
            ("A woman is slicing an onion.", "A woman is cutting onions.", 4.6),
            ("A man is riding a horse.", "A man is riding a white horse on an enclosed ground.", 3.8),
            ("A woman is playing the guitar.", "A man is playing guitar.", 2.4),
            ("Two dogs are wrestling and biting.", "Two dogs are playing by wrestling and biting.", 4.4),
            ("A child is riding a horse.", "A child is riding a pony.", 3.6),
            ("A man is smoking.", "A man is skating.", 0.5),
            ("A cat is playing with a ball.", "A dog is playing with a toy.", 2.0),
            ("A man is cutting up a chicken.", "A man is slicing a chicken.", 4.2),
            ("The woman is playing the violin.", "The young lady enjoys playing the violin.", 3.7),
            ("A man is eating food.", "A man is eating a piece of bread.", 3.0),
            ("A girl is styling her hair.", "A girl is brushing her hair.", 3.5),
            ("A baby is laughing.", "A baby is crying.", 1.6),
            ("A boy is running.", "A girl is running.", 2.2),
            ("A dog is catching a ball.", "A cat is ignoring everything.", 0.2),
            ("Two teams are competing.", "Two teams are playing a game.", 3.4),
            ("The sun is shining.", "It is a beautiful day.", 2.8),
            ("A man is driving a car.", "A man is driving a truck.", 2.5),
            ("A woman is reading a book.", "A woman is studying from a textbook.", 3.2),
            ("Someone is playing piano.", "A person practices on a keyboard.", 3.3),
            ("A cat is sleeping on a mat.", "The cat rests comfortably.", 3.6),
            ("A bird is flying over the ocean.", "A bird soars above the sea.", 4.5),
            ("Two men are fighting.", "Two men are boxing in a ring.", 3.0),
            ("A child is drawing a picture.", "A child is coloring with crayons.", 3.4),
            ("The chef is preparing dinner.", "A cook is making a meal.", 4.0),
            ("A dog is barking at a stranger.", "A dog is wagging its tail.", 1.0),
            ("Snow is falling.", "It is raining heavily.", 1.2),
            ("A student is studying math.", "A student is reading a novel.", 1.5),
            ("The train arrived at the station.", "The train pulled into the station.", 4.8),
        ],
        "metric": "spearman",
    },
}

CLASSIFICATION_DATASETS = {
    "intent": {
        "description": "Intent classification — 20 business intents",
        "samples": [
            ("How do I reset my password?", "account"),
            ("I need to change my email address.", "account"),
            ("Can I upgrade my subscription?", "billing"),
            ("What are the pricing plans?", "billing"),
            ("The app keeps crashing on startup.", "bug"),
            ("I see an error message when I log in.", "bug"),
            ("Can you add dark mode?", "feature_request"),
            ("It would be great to have export to PDF.", "feature_request"),
            ("What file formats do you support?", "general"),
            ("How does the search function work?", "general"),
            ("I want to cancel my account.", "cancellation"),
            ("Please delete all my data.", "cancellation"),
            ("The dashboard is loading very slowly.", "performance"),
            ("Reports take too long to generate.", "performance"),
            ("How do I integrate with Slack?", "integration"),
            ("Is there a Zapier connection available?", "integration"),
            ("I forgot my username.", "account"),
            ("My payment was declined.", "billing"),
            ("The mobile app shows a blank screen.", "bug"),
            ("Can you add multi-language support?", "feature_request"),
            ("What is your refund policy?", "billing"),
            ("How do I set up two-factor authentication?", "account"),
            ("The search results are not relevant.", "bug"),
            ("I would like bulk import functionality.", "feature_request"),
            ("What programming languages are supported?", "general"),
            ("I want to downgrade my plan.", "billing"),
            ("Image upload is not working.", "bug"),
            ("Please add a calendar view.", "feature_request"),
            ("How secure is my data?", "general"),
            ("API responses are very slow.", "performance"),
        ],
        "metric": "accuracy",
    },
}

CLUSTERING_DATASETS = {
    "topic_clusters": {
        "description": "Topic clustering — 6 topics, 30 items",
        "items": [
            ("Python programming tutorial for beginners", "tech"),
            ("JavaScript framework comparison React vs Vue", "tech"),
            ("Machine learning model training optimization", "tech"),
            ("Database indexing strategies for performance", "tech"),
            ("Cloud computing cost optimization AWS", "tech"),
            ("Premier League football scores this weekend", "sports"),
            ("NBA playoffs bracket predictions 2024", "sports"),
            ("Olympic swimming world record broken", "sports"),
            ("Tennis Grand Slam tournament results", "sports"),
            ("Formula 1 race qualifying times", "sports"),
            ("Stock market analysis quarterly earnings", "finance"),
            ("Cryptocurrency Bitcoin price prediction", "finance"),
            ("Federal reserve interest rate decision", "finance"),
            ("Real estate investment trust performance", "finance"),
            ("Personal finance budgeting strategies", "finance"),
            ("Homemade pasta recipe from scratch", "food"),
            ("Best chocolate cake baking tips", "food"),
            ("Healthy meal prep ideas for the week", "food"),
            ("Restaurant review fine dining experience", "food"),
            ("Vegetarian cooking substitutes guide", "food"),
            ("New movie releases this month action", "entertainment"),
            ("Music album review indie rock band", "entertainment"),
            ("Video game RPG walkthrough guide", "entertainment"),
            ("Television series season finale recap", "entertainment"),
            ("Broadway musical ticket prices NYC", "entertainment"),
            ("Climate change global warming effects", "science"),
            ("Space exploration Mars rover discovery", "science"),
            ("Quantum physics particle accelerator research", "science"),
            ("Genetic engineering CRISPR applications", "science"),
            ("Renewable energy solar panel efficiency", "science"),
        ],
        "metric": "v_measure",
    },
}


# --- Embedding client ---

async def get_embeddings(session, url, texts, semaphore, dimensions=0):
    """Get embeddings for a batch of texts via the served API."""
    results = []
    for i, text in enumerate(texts):
        async with semaphore:
            payload = {"input": text, "normalize": True, "modality": "text"}
            if dimensions > 0:
                payload["dimensions"] = dimensions
            try:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        emb = body.get("embedding", [])
                        if emb and all(isinstance(v, (int, float)) for v in emb):
                            results.append({"idx": i, "embedding": emb, "status": "ok"})
                        else:
                            results.append({"idx": i, "status": "error", "error": "invalid embedding"})
                    else:
                        results.append({"idx": i, "status": "error", "error": f"HTTP {resp.status}"})
            except Exception as exc:
                results.append({"idx": i, "status": "error", "error": str(exc)[:100]})
    return results


async def batch_embed(session, url, texts, concurrency=8, dimensions=0):
    """Embed a list of texts and return numpy array + success mask."""
    sem = asyncio.Semaphore(concurrency)
    results = await get_embeddings(session, url, texts, sem, dimensions=dimensions)
    embeddings = []
    valid_indices = []
    for r in sorted(results, key=lambda x: x["idx"]):
        if r["status"] == "ok":
            embeddings.append(r["embedding"])
            valid_indices.append(r["idx"])
    if not embeddings:
        return None, []
    return np.array(embeddings, dtype=np.float32), valid_indices


# --- Metrics (MTEB-aligned) ---

def ndcg_at_k(relevance_scores, k=10):
    """Compute nDCG@K from a list of relevance scores (graded or binary)."""
    relevance_scores = relevance_scores[:k]
    dcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevance_scores))
    ideal = sorted(relevance_scores, reverse=True)
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mean_average_precision(relevance_lists):
    """Compute MAP from a list of binary relevance lists."""
    aps = []
    for rels in relevance_lists:
        hits = 0
        precision_sum = 0.0
        for i, rel in enumerate(rels):
            if rel > 0:
                hits += 1
                precision_sum += hits / (i + 1)
        ap = precision_sum / max(hits, 1)
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0.0


def mrr_at_k(relevance_lists, k=10):
    """Compute MRR@K from a list of binary relevance lists."""
    rrs = []
    for rels in relevance_lists:
        for i, rel in enumerate(rels[:k]):
            if rel > 0:
                rrs.append(1.0 / (i + 1))
                break
        else:
            rrs.append(0.0)
    return sum(rrs) / len(rrs) if rrs else 0.0


def spearman_correlation(x, y):
    """Compute Spearman rank correlation between two lists."""
    from scipy.stats import spearmanr
    corr, pval = spearmanr(x, y)
    return float(corr), float(pval)


def v_measure_score(labels_true, labels_pred):
    """Compute V-measure (harmonic mean of homogeneity and completeness)."""
    from sklearn.metrics import v_measure_score as _v_measure
    return float(_v_measure(labels_true, labels_pred))


def cosine_similarity_matrix(a, b):
    """Compute cosine similarity matrix between two embedding matrices."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-8)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-8)
    return a_norm @ b_norm.T


# --- Evaluation suites ---

async def eval_retrieval(session, url, concurrency=8, dimensions=0):
    """Evaluate retrieval quality using nDCG@10, MAP, MRR@10."""
    log.info("## Retrieval Quality (nDCG@10, MAP, MRR@10)")
    results = {}

    for ds_name, ds in RETRIEVAL_DATASETS.items():
        log.info(f"  {ds_name}: {ds['description']}")

        queries = ds["queries"]
        corpus = ds["corpus"]
        assert len(queries) == len(corpus), f"{ds_name}: queries and corpus must be same length"

        q_embs, q_valid = await batch_embed(session, url, queries, concurrency, dimensions)
        c_embs, c_valid = await batch_embed(session, url, corpus, concurrency, dimensions)

        if q_embs is None or c_embs is None:
            results[ds_name] = {"status": "error", "reason": "embedding failed"}
            continue

        sim = cosine_similarity_matrix(q_embs, c_embs)

        ndcg_scores = []
        map_rels = []
        for qi, q_idx in enumerate(q_valid):
            ranked = np.argsort(sim[qi])[::-1][:10]
            rels = [1.0 if c_valid[int(r)] == q_idx else 0.0 for r in ranked]
            ndcg_scores.append(ndcg_at_k(rels, k=10))
            map_rels.append(rels)

        results[ds_name] = {
            "status": "ok",
            "ndcg_at_10": round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else 0,
            "map": round(mean_average_precision(map_rels), 4),
            "mrr_at_10": round(mrr_at_k(map_rels, k=10), 4),
            "queries": len(q_valid),
            "corpus": len(c_valid),
        }
        log.info(f"    nDCG@10={results[ds_name]['ndcg_at_10']:.4f}  "
                 f"MAP={results[ds_name]['map']:.4f}  "
                 f"MRR@10={results[ds_name]['mrr_at_10']:.4f}")

    return results


async def eval_sts(session, url, concurrency=8, dimensions=0):
    """Evaluate semantic textual similarity using Spearman correlation."""
    log.info("## Semantic Textual Similarity (Spearman)")
    results = {}

    for ds_name, ds in STS_DATASETS.items():
        log.info(f"  {ds_name}: {ds['description']}")
        pairs = ds["pairs"]
        texts_a = [p[0] for p in pairs]
        texts_b = [p[1] for p in pairs]
        gold_scores = [p[2] for p in pairs]

        embs_a, valid_a = await batch_embed(session, url, texts_a, concurrency, dimensions)
        embs_b, valid_b = await batch_embed(session, url, texts_b, concurrency, dimensions)

        if embs_a is None or embs_b is None:
            results[ds_name] = {"status": "error", "reason": "embedding failed"}
            continue

        common = set(valid_a) & set(valid_b)
        if len(common) < 5:
            results[ds_name] = {"status": "error", "reason": "too few valid pairs"}
            continue

        idx_map_a = {v: i for i, v in enumerate(valid_a)}
        idx_map_b = {v: i for i, v in enumerate(valid_b)}

        cosines = []
        golds = []
        for idx in sorted(common):
            a_vec = embs_a[idx_map_a[idx]]
            b_vec = embs_b[idx_map_b[idx]]
            cos = float(np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec) + 1e-8))
            cosines.append(cos)
            golds.append(gold_scores[idx])

        try:
            corr, pval = spearman_correlation(cosines, golds)
        except Exception as e:
            results[ds_name] = {"status": "error", "reason": str(e)[:100]}
            continue

        results[ds_name] = {
            "status": "ok",
            "spearman": round(corr, 4),
            "p_value": round(pval, 6),
            "pairs_evaluated": len(common),
        }
        log.info(f"    Spearman={corr:.4f}  (p={pval:.4e}, n={len(common)})")

    return results


async def eval_classification(session, url, concurrency=8, dimensions=0):
    """Evaluate embedding quality via linear probe classification accuracy."""
    log.info("## Classification (Linear Probe Accuracy)")
    results = {}

    for ds_name, ds in CLASSIFICATION_DATASETS.items():
        log.info(f"  {ds_name}: {ds['description']}")
        texts = [s[0] for s in ds["samples"]]
        labels = [s[1] for s in ds["samples"]]

        embs, valid = await batch_embed(session, url, texts, concurrency, dimensions)
        if embs is None or len(valid) < 10:
            results[ds_name] = {"status": "error", "reason": "too few embeddings"}
            continue

        valid_labels = [labels[i] for i in valid]
        unique_labels = sorted(set(valid_labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        y = np.array([label_to_idx[l] for l in valid_labels])

        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=100, random_state=42)
        try:
            scores = cross_val_score(clf, embs, y, cv=min(5, len(unique_labels)), scoring="accuracy")
            accuracy = float(np.mean(scores))
        except Exception as e:
            results[ds_name] = {"status": "error", "reason": str(e)[:100]}
            continue

        results[ds_name] = {
            "status": "ok",
            "accuracy": round(accuracy, 4),
            "cv_std": round(float(np.std(scores)), 4),
            "classes": len(unique_labels),
            "samples": len(valid),
        }
        log.info(f"    Accuracy={accuracy:.4f} ± {np.std(scores):.4f} ({len(unique_labels)} classes)")

    return results


async def eval_clustering(session, url, concurrency=8, dimensions=0):
    """Evaluate embedding quality via K-means clustering V-measure."""
    log.info("## Clustering (V-measure)")
    results = {}

    for ds_name, ds in CLUSTERING_DATASETS.items():
        log.info(f"  {ds_name}: {ds['description']}")
        texts = [item[0] for item in ds["items"]]
        true_labels = [item[1] for item in ds["items"]]

        embs, valid = await batch_embed(session, url, texts, concurrency, dimensions)
        if embs is None or len(valid) < 10:
            results[ds_name] = {"status": "error", "reason": "too few embeddings"}
            continue

        valid_labels = [true_labels[i] for i in valid]
        n_clusters = len(set(valid_labels))

        from sklearn.cluster import KMeans

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            pred_labels = kmeans.fit_predict(embs)
            vm = v_measure_score(valid_labels, pred_labels)
        except Exception as e:
            results[ds_name] = {"status": "error", "reason": str(e)[:100]}
            continue

        results[ds_name] = {
            "status": "ok",
            "v_measure": round(vm, 4),
            "n_clusters": n_clusters,
            "samples": len(valid),
        }
        log.info(f"    V-measure={vm:.4f} ({n_clusters} clusters, {len(valid)} samples)")

    return results


async def eval_dimension_reduction(session, url, dimensions_list, concurrency=8):
    """Evaluate quality at each Matryoshka truncation level."""
    log.info(f"\n## Dimension Reduction Curves ({dimensions_list})")
    results = {}

    for dim in dimensions_list:
        log.info(f"\n--- Dimension: {dim} ---")
        dim_results = {}

        sts = await eval_sts(session, url, concurrency, dimensions=dim)
        dim_results["sts"] = sts

        cls = await eval_classification(session, url, concurrency, dimensions=dim)
        dim_results["classification"] = cls

        cluster = await eval_clustering(session, url, concurrency, dimensions=dim)
        dim_results["clustering"] = cluster

        summary = {}
        for ds_name, ds_result in sts.items():
            if ds_result.get("status") == "ok":
                summary["spearman"] = ds_result["spearman"]
        for ds_name, ds_result in cls.items():
            if ds_result.get("status") == "ok":
                summary["accuracy"] = ds_result["accuracy"]
        for ds_name, ds_result in cluster.items():
            if ds_result.get("status") == "ok":
                summary["v_measure"] = ds_result["v_measure"]

        dim_results["summary"] = summary
        results[str(dim)] = dim_results

        log.info(f"  dim={dim}: " + ", ".join(f"{k}={v:.4f}" for k, v in summary.items()))

    return results


async def main():
    parser = argparse.ArgumentParser(description="Embedding quality benchmark for highperfasr")
    parser.add_argument("--server", default="http://localhost:8000", help="Server base URL")
    parser.add_argument(
        "--suite", default="all",
        choices=["all", "quick", "retrieval", "sts", "classification", "clustering", "dimensions"],
        help="Which evaluation suite to run (default: all)",
    )
    parser.add_argument("--endpoint", default="/v1/embeddings", help="Embedding endpoint path")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent requests (default: 8)")
    parser.add_argument(
        "--dimensions", default="768,512,256,128",
        help="Comma-separated dimension levels for Matryoshka eval (default: 768,512,256,128)",
    )
    parser.add_argument("--output", default="/tmp/bench_embed_quality.json", help="Output JSON path")
    parser.add_argument("--gpu", default=None, metavar="NAME", help="Override GPU name in report")
    args = parser.parse_args()

    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    embed_url = f"{args.server.rstrip('/')}{endpoint}"

    dim_list = [int(d) for d in args.dimensions.split(",")]

    server_info = detect_server(args.server)
    log_preflight_summary(server_info, mode_requested="embedding")

    log.info(f"Embedding endpoint: {embed_url}")
    log.info(f"Evaluation suite: {args.suite}")

    report = {
        "benchmark": "embedding_quality",
        "suite": args.suite,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "server": args.server,
    }
    if args.gpu:
        report["gpu"] = args.gpu

    connector = aiohttp.TCPConnector(limit=max(args.concurrency + 16, 64))
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        log.info("Warming up...")
        sem = asyncio.Semaphore(4)
        warmup_results = await get_embeddings(
            session, embed_url, ["warmup text"] * 5, sem
        )
        ok = sum(1 for r in warmup_results if r["status"] == "ok")
        if ok == 0:
            log.error("Warmup failed — aborting")
            report["error"] = "warmup failed"
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            return

        for r in warmup_results:
            if r["status"] == "ok":
                report["embedding_dimensions"] = len(r["embedding"])
                break

        suites_to_run = []
        if args.suite == "all":
            suites_to_run = ["retrieval", "sts", "classification", "clustering", "dimensions"]
        elif args.suite == "quick":
            suites_to_run = ["sts", "classification", "clustering"]
        else:
            suites_to_run = [args.suite]

        t0 = time.monotonic()

        if "retrieval" in suites_to_run:
            report["retrieval"] = await eval_retrieval(session, embed_url, args.concurrency)

        if "sts" in suites_to_run:
            report["sts"] = await eval_sts(session, embed_url, args.concurrency)

        if "classification" in suites_to_run:
            report["classification"] = await eval_classification(session, embed_url, args.concurrency)

        if "clustering" in suites_to_run:
            report["clustering"] = await eval_clustering(session, embed_url, args.concurrency)

        if "dimensions" in suites_to_run:
            report["dimension_reduction"] = await eval_dimension_reduction(
                session, embed_url, dim_list, args.concurrency
            )

        report["total_time_s"] = round(time.monotonic() - t0, 2)

    report["quality_summary"] = build_quality_summary(report)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nReport written to {args.output}")

    print_summary(report)


def build_quality_summary(report):
    """Build a compact quality summary from evaluation results."""
    summary = {}

    retrieval = report.get("retrieval", {})
    ndcg_scores = []
    for ds_name, ds_result in retrieval.items():
        if isinstance(ds_result, dict) and ds_result.get("status") == "ok":
            ndcg_scores.append(ds_result["ndcg_at_10"])
    if ndcg_scores:
        summary["retrieval_ndcg_at_10"] = round(np.mean(ndcg_scores), 4)

    sts = report.get("sts", {})
    for ds_name, ds_result in sts.items():
        if isinstance(ds_result, dict) and ds_result.get("status") == "ok":
            summary["sts_spearman"] = ds_result["spearman"]

    cls = report.get("classification", {})
    for ds_name, ds_result in cls.items():
        if isinstance(ds_result, dict) and ds_result.get("status") == "ok":
            summary["classification_accuracy"] = ds_result["accuracy"]

    clustering = report.get("clustering", {})
    for ds_name, ds_result in clustering.items():
        if isinstance(ds_result, dict) and ds_result.get("status") == "ok":
            summary["clustering_v_measure"] = ds_result["v_measure"]

    dim_reduction = report.get("dimension_reduction", {})
    if dim_reduction:
        dim_curve = {}
        for dim, dim_data in dim_reduction.items():
            if isinstance(dim_data, dict) and "summary" in dim_data:
                dim_curve[dim] = dim_data["summary"]
        if dim_curve:
            summary["dimension_curve"] = dim_curve

    return summary


def print_summary(report):
    """Print human-readable summary."""
    print(f"\n{'='*60}")
    print(f"## Embedding Quality Summary")
    print(f"Suite: {report.get('suite', 'unknown')}")
    print(f"Time: {report.get('total_time_s', 0):.1f}s")
    print(f"Dimensions: {report.get('embedding_dimensions', 'unknown')}")

    summary = report.get("quality_summary", {})

    if "retrieval_ndcg_at_10" in summary:
        print(f"\n  Retrieval nDCG@10: {summary['retrieval_ndcg_at_10']:.4f}")

    if "sts_spearman" in summary:
        print(f"  STS Spearman:      {summary['sts_spearman']:.4f}")

    if "classification_accuracy" in summary:
        print(f"  Classification:    {summary['classification_accuracy']:.4f}")

    if "clustering_v_measure" in summary:
        print(f"  Clustering V:      {summary['clustering_v_measure']:.4f}")

    dim_curve = summary.get("dimension_curve", {})
    if dim_curve:
        print(f"\n  Dimension Reduction Curve:")
        print(f"  {'Dim':>6} | {'Spearman':>9} | {'Accuracy':>9} | {'V-measure':>9}")
        print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
        for dim in sorted(dim_curve.keys(), key=lambda x: int(x), reverse=True):
            vals = dim_curve[dim]
            sp = f"{vals.get('spearman', 0):.4f}" if "spearman" in vals else "   —"
            acc = f"{vals.get('accuracy', 0):.4f}" if "accuracy" in vals else "   —"
            vm = f"{vals.get('v_measure', 0):.4f}" if "v_measure" in vals else "   —"
            print(f"  {dim:>6} | {sp:>9} | {acc:>9} | {vm:>9}")

    retrieval = report.get("retrieval", {})
    if retrieval:
        print(f"\n  Retrieval Detail:")
        for ds_name, ds_result in retrieval.items():
            if isinstance(ds_result, dict) and ds_result.get("status") == "ok":
                print(f"    {ds_name}: nDCG@10={ds_result['ndcg_at_10']:.4f}  "
                      f"MAP={ds_result['map']:.4f}  MRR@10={ds_result['mrr_at_10']:.4f}")

    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
