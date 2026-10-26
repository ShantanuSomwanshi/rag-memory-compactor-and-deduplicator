"""Side-by-side retrieval demo: the same query against the store before and
after compaction.

Run with:  streamlit run app.py

The "before" index is not a saved copy - it is reconstructed from the live store
plus the archive, which is only possible because every merge archived its
originals together with their embedding vectors.
"""

from __future__ import annotations

import html
import itertools
import json
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ragcompactor.backends import VectorStore
from ragcompactor.benchmark import rebuild_before_store
from ragcompactor.compactor import Compactor
from ragcompactor.core import Chunk, CompactorConfig, count_tokens

st.set_page_config(page_title="RAG Memory Compactor", layout="wide")

# Projected on a screen at the back of a room: bigger type, stronger contrast,
# less of Streamlit's default grey.
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { font-size: 16px; }
      .block-container { padding-top: 2.2rem; max-width: 1500px; }
      h1 { font-size: 2.2rem !important; }
      h2 { font-size: 1.5rem !important; }
      h3 { font-size: 1.15rem !important; }
      [data-testid="stMetricValue"] { font-size: 1.9rem; }
      [data-testid="stMetricLabel"] { font-size: 0.85rem; font-weight: 600; }
      .rc-text { font-size: 14.5px; line-height: 1.55; color: inherit; }
      .rc-meta { font-size: 12.5px; color: #5A6A6A; }
      .rc-dup  { background: #F6E3D8; color: #3B241C !important;
                 border-left: 3px solid #A3502E; padding: 2px 6px;
                 border-radius: 2px; box-decoration-break: clone;
                 -webkit-box-decoration-break: clone; }
      .rc-ans  { font-size: 15px; line-height: 1.7; min-width: 0;
                 max-width: 100%; overflow-wrap: anywhere; }
    </style>
    """,
    unsafe_allow_html=True,
)

QUERIES_FILE = Path("queries.txt")

FALLBACK_QUERIES = [
    "What conditions must local governments meet to be eligible for grants?",
    "How are mitigation funds and disaster risk assessment institutionalised?",
    "What criteria and weights are used for horizontal devolution?",
]

ANSWER_PROMPT = (
    "Answer the question using ONLY the numbered context passages below. "
    "If the passages do not contain the answer, say so plainly. "
    "Be specific: cite figures and names exactly as they appear. Keep it under 120 words."
)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading store and embedding model...")
def load() -> tuple[Compactor, VectorStore, VectorStore]:
    compactor = Compactor(CompactorConfig.load("ragcompactor.json"))
    return compactor, rebuild_before_store(compactor), compactor.store


def load_queries() -> list[str]:
    if QUERIES_FILE.exists():
        found = [
            q.strip()
            for q in QUERIES_FILE.read_text(encoding="utf-8").splitlines()
            if q.strip() and not q.strip().startswith("#")
        ]
        if found:
            return found
    return FALLBACK_QUERIES


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def retrieve(store: VectorStore, embedder, query: str, k: int) -> list[dict]:
    vector = embedder.embed([query])[0]
    hits = []
    for rank, (chunk_id, score) in enumerate(store.query(vector, k=k), 1):
        chunk = store.get(chunk_id)
        if chunk is None:
            continue
        hits.append(
            {
                "rank": rank,
                "id": chunk_id,
                "score": float(score),
                "chunk": chunk,
                "vector": store.get_vector(chunk_id),
                "tokens": count_tokens(chunk.text, "gpt-4o-mini"),
            }
        )
    return hits


def mark_duplicates(hits: list[dict], threshold: float) -> None:
    """Flag each hit that repeats material already present higher up."""
    for hit in hits:
        hit["duplicate_of"] = None
    for i, j in itertools.combinations(range(len(hits)), 2):
        vi, vj = hits[i]["vector"], hits[j]["vector"]
        if vi is None or vj is None:
            continue
        sim = float(np.dot(vi, vj))
        if sim >= threshold and hits[j]["duplicate_of"] is None:
            hits[j]["duplicate_of"] = (hits[i]["rank"], sim)


def source_label(chunk: Chunk) -> str:
    if chunk.is_merged:
        names = [Path(s).name for s in chunk.metadata.get("merged_sources", []) if s]
        return " + ".join(names) if names else "merged"
    return f"{Path(chunk.source).name if chunk.source else '(no source)'} #{chunk.ordinal}"


# ---------------------------------------------------------------------------
# answers, and redundancy inside them
# ---------------------------------------------------------------------------


def sentences_of(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]


def answer_redundancy(text: str, embedder, threshold: float) -> tuple[list[str], dict]:
    """Find statements in an answer that repeat an earlier statement.

    Deliberately the same machinery the compactor uses on the corpus: embed,
    then compare. A duplicate in the retrieved context tends to reappear as a
    duplicate in the answer - often introduced as if it were something new.
    """
    sents = sentences_of(text)
    repeats: dict[int, tuple[int, float]] = {}
    if len(sents) < 2:
        return sents, repeats
    vecs = embedder.embed(sents)
    for i, j in itertools.combinations(range(len(sents)), 2):
        sim = float(np.dot(vecs[i], vecs[j]))
        if sim >= threshold and j not in repeats:
            repeats[j] = (i, sim)
    return sents, repeats


def duplicate_answer_sources(
    text: str, hits: list[dict], embedder, threshold: float
) -> dict[int, tuple[int, float]]:
    """Match answer statements to statements in duplicate retrieved chunks."""
    answer_sents = sentences_of(text)
    source_sents = [
        (hit["rank"], sentence)
        for hit in hits
        if hit["duplicate_of"] is not None
        for sentence in all_sentences(hit["chunk"].text)
        if len(sentence) > 25
    ]
    if not answer_sents or not source_sents:
        return {}

    vectors = embedder.embed(
        answer_sents + [sentence for _rank, sentence in source_sents]
    )
    similarities = vectors[: len(answer_sents)] @ vectors[len(answer_sents) :].T
    matches = {}
    for idx, row in enumerate(similarities):
        source_idx = int(np.argmax(row))
        similarity = float(row[source_idx])
        if similarity >= threshold:
            matches[idx] = (source_sents[source_idx][0], similarity)
    return matches


def render_answer(text: str, embedder, threshold: float, hits: list[dict]) -> int:
    sents, repeats = answer_redundancy(text, embedder, threshold)
    duplicate_sources = duplicate_answer_sources(text, hits, embedder, threshold)
    if not sents:
        st.write(text)
        return 0
    parts = []
    for idx, sentence in enumerate(sents):
        title_parts = []
        if idx in repeats:
            src, sim = repeats[idx]
            title_parts.append(
                f"repeats answer statement {src + 1} (similarity {sim:.2f})"
            )
        if idx in duplicate_sources:
            rank, sim = duplicate_sources[idx]
            title_parts.append(
                f"matches duplicate retrieved chunk #{rank} (similarity {sim:.2f})"
            )
        safe_sentence = html.escape(sentence, quote=False)
        if title_parts:
            title = html.escape("; ".join(title_parts), quote=True)
            parts.append(
                f"<span class='rc-dup' title='{title}'>{safe_sentence}</span>"
            )
        else:
            parts.append(safe_sentence)
    st.markdown(f"<div class='rc-ans'>{' '.join(parts)}</div>", unsafe_allow_html=True)
    return len(repeats.keys() | duplicate_sources.keys())


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{h['rank']}] {' '.join(h['chunk'].text.split())}" for h in hits)


def generate_answer(question: str, hits: list[dict], cfg: CompactorConfig) -> dict:
    import litellm

    response = litellm.completion(
        model=cfg.llm_model,
        messages=[
            {"role": "system", "content": ANSWER_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}",
            },
        ],
        temperature=0.0,
        max_tokens=320,
        timeout=cfg.llm_timeout,
        num_retries=cfg.llm_num_retries,
    )
    usage = getattr(response, "usage", None)
    return {
        "text": (response.choices[0].message.content or "").strip(),
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


# ---------------------------------------------------------------------------
# merge provenance - where each sentence of a merged chunk came from
# ---------------------------------------------------------------------------

PROV_COLORS = [
    ("#D7EBE8", "#06433F"),
    ("#DCE4F2", "#24456B"),
    ("#E2EEDB", "#3A5C25"),
    ("#EDE1F2", "#52336C"),
    ("#F6E9D5", "#6E5010"),
]
PROV_UNMATCHED = ("#F8DADA", "#8E2626")
ANSWER_THRESHOLD = 0.68
PROV_THRESHOLD = 0.60


def all_sentences(text: str) -> list[str]:
    """Every sentence, including short ones - nothing is dropped from display."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def merge_coverage(originals: list[Chunk], merged_text: str, embedder, threshold: float):
    """For every sentence of every original, did it survive into the merge?

    The mirror of :func:`provenance`. That one asks "where did this merged
    sentence come from"; this one asks "did this original sentence get carried
    over" - and a No here is the thing that actually matters, because it is
    information the store no longer holds.
    """
    merged_sents = [s for s in all_sentences(merged_text) if len(s) > 25]
    per_original = []

    if not merged_sents:
        return [[(s, "short", 0.0) for s in all_sentences(c.text)] for c in originals]

    merged_vecs = embedder.embed(merged_sents)
    for chunk in originals:
        sents = all_sentences(chunk.text)
        long_idx = [i for i, sentence in enumerate(sents) if len(sentence) > 25]
        rows = [(sentence, "short", 0.0) for sentence in sents]
        if long_idx:
            vecs = embedder.embed([sents[i] for i in long_idx])
            sims = vecs @ merged_vecs.T
            for row, i in enumerate(long_idx):
                best = float(np.max(sims[row]))
                rows[i] = (sents[i], best >= threshold, best)
        per_original.append(rows)
    return per_original


def render_coverage(rows, origin_idx: int) -> tuple[int, int]:
    """Paint an original: carried-over sentences in its colour, dropped ones red."""
    bg_keep, fg_keep = PROV_COLORS[origin_idx % len(PROV_COLORS)]
    bg_drop, fg_drop = PROV_UNMATCHED
    parts, kept, dropped = [], 0, 0
    for sentence, state, sim in rows:
        if state == "short":
            parts.append(f"<span style='opacity:.75'>{sentence}</span>")
            continue
        if state:
            kept += 1
            parts.append(
                f"<span style='background:{bg_keep};color:{fg_keep};padding:1px 4px;"
                f"border-radius:2px' title='carried into the merge (similarity "
                f"{sim:.2f})'>{sentence}</span>"
            )
        else:
            dropped += 1
            parts.append(
                f"<span style='background:{bg_drop};color:{fg_drop};padding:1px 4px;"
                f"border-radius:2px' title='NOT carried into the merge (best match "
                f"{sim:.2f})'>{sentence}</span>"
            )
    st.markdown(f"<div class='rc-text'>{' '.join(parts)}</div>", unsafe_allow_html=True)
    return kept, dropped


def provenance(merged_text: str, originals: list[Chunk], embedder, threshold: float):
    """Trace every sentence of a merged chunk back to the original it came from.

    Each sentence of the merge is matched against every sentence of every
    archived original; the closest one wins. A sentence matching nothing closely
    is the interesting case - it means the model wrote something the sources did
    not say, which is the failure mode summarization-based compaction has to be
    checked for.
    """
    merged_sents = all_sentences(merged_text)
    # Short fragments ("See Table 8.8.") carry too little signal to match on, but
    # they must still be displayed - dropping them would make the merged chunk
    # look shorter than it is, in the very view meant to prove nothing was lost.
    traceable = [i for i, s in enumerate(merged_sents) if len(s) > 25]

    origin_sents: list[str] = []
    origin_of: list[int] = []
    for idx, chunk in enumerate(originals):
        for sentence in all_sentences(chunk.text):
            if len(sentence) > 25:
                origin_sents.append(sentence)
                origin_of.append(idx)

    if not traceable or not origin_sents:
        return [(s, "short", 0.0) for s in merged_sents]

    merged_vecs = embedder.embed([merged_sents[i] for i in traceable])
    origin_vecs = embedder.embed(origin_sents)
    sims = merged_vecs @ origin_vecs.T

    best_for = {}
    for row, i in enumerate(traceable):
        j = int(np.argmax(sims[row]))
        best = float(sims[row][j])
        best_for[i] = (origin_of[j] if best >= threshold else None, best)

    return [
        (sentence, *best_for.get(i, ("short", 0.0)))
        for i, sentence in enumerate(merged_sents)
    ]


def render_provenance(traced) -> int:
    """Paint merged text, each sentence tinted by source. Returns untraced count."""
    parts, untraced = [], 0
    for sentence, origin, sim in traced:
        if origin == "short":
            parts.append(f"<span style='opacity:.75'>{sentence}</span>")
            continue
        if origin is None:
            bg, fg = PROV_UNMATCHED
            tip = f"no close match in any original (best {sim:.2f})"
            untraced += 1
        else:
            bg, fg = PROV_COLORS[origin % len(PROV_COLORS)]
            tip = f"from original {origin + 1} (similarity {sim:.2f})"
        parts.append(
            f"<span style='background:{bg};color:{fg};padding:1px 4px;"
            f"border-radius:2px' title='{tip}'>{sentence}</span>"
        )
    st.markdown(f"<div class='rc-text'>{' '.join(parts)}</div>", unsafe_allow_html=True)
    return untraced


def render_prov_legend(n_originals: int, untraced: int) -> None:
    chips = []
    for i in range(n_originals):
        bg, fg = PROV_COLORS[i % len(PROV_COLORS)]
        chips.append(
            f"<span style='background:{bg};color:{fg};padding:2px 9px;border-radius:3px;"
            f"font-size:11.5px;font-weight:700;margin-right:6px'>original {i + 1}</span>"
        )
    bg, fg = PROV_UNMATCHED
    chips.append(
        f"<span style='background:{bg};color:{fg};padding:2px 9px;border-radius:3px;"
        f"font-size:11.5px;font-weight:700'>not traceable &middot; {untraced}</span>"
    )
    st.markdown("".join(chips), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

BADGE = (
    "<span style='background:{bg};color:{fg};padding:2px 8px;border-radius:3px;"
    "font-size:11.5px;font-weight:700;letter-spacing:.04em;margin-right:6px;'>{label}</span>"
)


def render_hit(hit: dict, promoted_ids: set[str]) -> None:
    chunk = hit["chunk"]
    badges = ""
    if hit["duplicate_of"] is not None:
        rank, sim = hit["duplicate_of"]
        badges += BADGE.format(
            bg="#F3DCCE", fg="#8A3F1C", label=f"DUPLICATE OF #{rank} · {sim:.3f}"
        )
    if chunk.is_merged:
        n = len(chunk.metadata.get("merged_from", []))
        badges += BADGE.format(bg="#CFE7E3", fg="#07514F", label=f"MERGED FROM {n}")
    if hit["id"] in promoted_ids:
        badges += BADGE.format(bg="#F7E2A8", fg="#6B4E06", label="PROMOTED")

    with st.container(border=True):
        head, meta = st.columns([3, 2])
        head.markdown(f"**#{hit['rank']}** &nbsp; `{hit['score']:.3f}`")
        meta.markdown(
            f"<div class='rc-meta' style='text-align:right'>"
            f"{source_label(chunk)} · {hit['tokens']} tok</div>",
            unsafe_allow_html=True,
        )
        if badges:
            st.markdown(badges, unsafe_allow_html=True)
        text = " ".join(chunk.text.split())
        st.markdown(
            f"<div class='rc-text'>{text[:400]}{'...' if len(text) > 400 else ''}</div>",
            unsafe_allow_html=True,
        )
        if chunk.is_merged:
            merge_id = MERGE_BY_CHUNK.get(hit["id"])
            if merge_id:
                with st.expander("where each sentence came from"):
                    originals = [c for c, _v in compactor.archive.originals_for(merge_id)]
                    traced = provenance(
                        chunk.text, originals, compactor.embedder, PROV_THRESHOLD
                    )
                    untraced = render_provenance(traced)
                    st.write("")
                    render_prov_legend(len(originals), untraced)
        if len(text) > 400:
            with st.expander("full text"):
                st.write(text)


def render_retrieval_graph(hits: list[dict], after_store: VectorStore) -> None:
    hits_by_rank = {hit["rank"]: hit for hit in hits}
    pairs = []
    for duplicate in hits:
        duplicate_of = duplicate["duplicate_of"]
        if duplicate_of is None:
            continue
        primary = hits_by_rank.get(duplicate_of[0])
        if primary is None:
            continue

        merge = next(
            (
                item
                for item in merges
                if primary["id"] in item.original_ids
                and duplicate["id"] in item.original_ids
            ),
            None,
        )
        result = after_store.get(merge.merged_chunk_id) if merge else None
        if result is not None:
            result_label = source_label(result)
            result_text = result.text
        elif merge is not None:
            result_label = f"Merged result · {merge.merge_id}"
            result_text = merge.merged_text
        else:
            result_label = f"Deduplicated result · kept #{primary['rank']}"
            result_text = primary["chunk"].text

        label = (
            f"#{primary['rank']} {source_label(primary['chunk'])} + "
            f"#{duplicate['rank']} {source_label(duplicate['chunk'])}"
        )
        pairs.append(
            (label, primary, duplicate, result_label, result_text, merge is not None)
        )

    st.subheader("Retrieved chunks → compaction result")
    if not pairs:
        st.info("No duplicate chunk pair was found in this retrieval.")
        return

    selected = st.selectbox(
        "Duplicate pair to visualize",
        options=range(len(pairs)),
        format_func=lambda index: pairs[index][0],
        key="retrieval_graph_pair",
    )
    _label, primary, duplicate, result_label, result_text, was_merged = pairs[selected]
    if was_merged:
        st.caption("This pair was archived together in an actual compaction merge.")
    else:
        st.caption(
            "These retrieved chunks matched the duplicate threshold; the higher-ranked "
            "chunk is shown as the retained deduplication result."
        )

    dot = f"""
    digraph retrieval {{
      graph [rankdir=LR, bgcolor="transparent", nodesep=0.55, ranksep=0.8,
             pad=0.2, tooltip=""];
      node [shape=box, style="rounded,filled", fontname="sans-serif",
            fontsize=12, margin="0.16,0.10", color="#718096",
            fillcolor="#202733", fontcolor="#F4F6F8"];
      edge [color="#8492A6", penwidth=1.6, arrowsize=0.8];
      source_a [label={json.dumps(source_label(primary["chunk"]), ensure_ascii=False)},
                tooltip={json.dumps(primary["chunk"].text, ensure_ascii=False)}];
      source_b [label={json.dumps(source_label(duplicate["chunk"]), ensure_ascii=False)},
                tooltip={json.dumps(duplicate["chunk"].text, ensure_ascii=False)}];
      result [label={json.dumps(result_label, ensure_ascii=False)},
              tooltip={json.dumps(result_text, ensure_ascii=False)},
              fillcolor="#244A43", color="#4F9B82"];
      source_a -> result;
      source_b -> result;
    }}
    """
    st.graphviz_chart(dot, use_container_width=True)


# ---------------------------------------------------------------------------
# app shell
# ---------------------------------------------------------------------------

st.title("RAG Memory Compactor")
st.caption(
    "The same query against the index before and after redundant chunks were merged. "
    "The 'before' index is reconstructed from the archive, not stored separately."
)

try:
    compactor, before_store, after_store = load()
except Exception as exc:  # pragma: no cover - surfaced in the UI
    st.error(f"Could not open the store: {exc}")
    st.stop()

cfg = compactor.config
merges = compactor.archive.list_merges(include_undone=False)
MERGE_BY_CHUNK = {m.merged_chunk_id: m.merge_id for m in merges}

with st.sidebar:
    st.subheader("Index")
    st.metric("Before compaction", f"{before_store.count()} chunks")
    st.metric("After compaction", f"{after_store.count()} chunks")
    st.caption(f"{len(merges)} merges applied")
    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("chunks retrieved (top-k)", 3, 10, 5)
    want_answers = st.checkbox("Generate answers from both contexts", value=True)
    st.divider()
    st.subheader("Thresholds")
    st.caption(f"**discovery ≥ {cfg.similarity_threshold}** — link possible duplicates")
    st.caption(f"**merge gate ≥ {cfg.validation_threshold}** — every pair must clear this")
    st.caption(f"embedder `{cfg.embedding_model}`")
    st.caption(f"model `{cfg.llm_model}`")

if not merges:
    st.warning(
        "No merges recorded, so both sides are identical. "
        "Run `ragcompactor compact --throttle 12` first."
    )

tab_compare, tab_merges, tab_map = st.tabs(
    ["Retrieval comparison", "Merge inspector", "Embedding map"]
)

# ---------------------------------------------------------------------------
# tab 1 - retrieval comparison
# ---------------------------------------------------------------------------

with tab_compare:
    queries = load_queries()
    choice = st.selectbox("Question", queries + ["Type my own..."])
    question = (
        st.text_input("Your question", placeholder="Ask anything about the corpus")
        if choice == "Type my own..."
        else choice
    )

    if st.button("Retrieve", type="primary") and question:
        before_hits = retrieve(before_store, compactor.embedder, question, top_k)
        after_hits = retrieve(after_store, compactor.embedder, question, top_k)
        mark_duplicates(before_hits, cfg.validation_threshold)
        mark_duplicates(after_hits, cfg.validation_threshold)

        before_ids = {h["id"] for h in before_hits}
        promoted = {
            h["id"]
            for h in after_hits
            if h["id"] not in before_ids and not h["chunk"].is_merged
        }
        before_tokens = sum(h["tokens"] for h in before_hits)
        after_tokens = sum(h["tokens"] for h in after_hits)
        redundant = sum(1 for h in before_hits if h["duplicate_of"] is not None)
        src_before = len({source_label(h["chunk"]).split(" #")[0] for h in before_hits})
        src_after = len({source_label(h["chunk"]).split(" #")[0] for h in after_hits})

        m = st.columns(4)
        m[0].metric("Context tokens", after_tokens, delta=after_tokens - before_tokens)
        m[1].metric("Redundant hits", f"{redundant} → 0" if redundant else "0")
        m[2].metric("Distinct sources", src_after, delta=src_after - src_before)
        m[3].metric("Newly reaching the model", len(promoted))

        if redundant:
            st.info(
                f"**{redundant} of the {top_k} retrieved chunks repeated material already "
                "present higher up.** Those slots cost tokens and returned nothing new."
            )
        if promoted:
            st.success(
                f"**{len(promoted)} chunk(s) now reach the model that did not before.** "
                "Freeing the duplicated slots let genuinely new material into the context."
            )

        left, right = st.columns(2)
        with left:
            st.subheader("Before compaction")
            st.caption(f"{before_store.count()} indexed · {before_tokens} tokens retrieved")
            for hit in before_hits:
                render_hit(hit, set())
        with right:
            st.subheader("After compaction")
            st.caption(f"{after_store.count()} indexed · {after_tokens} tokens retrieved")
            for hit in after_hits:
                render_hit(hit, promoted)

        render_retrieval_graph(before_hits, after_store)

        if want_answers:
            st.divider()
            st.subheader("Answers generated from each context")
            st.caption(
                "Shaded statements either repeat an earlier statement in the answer or "
                "match a retrieved chunk marked as a duplicate."
            )
            a, b = st.columns(2)
            for col, hits, label in ((a, before_hits, "before"), (b, after_hits, "after")):
                with col:
                    st.markdown(f"**From the {label} index**")
                    with st.spinner("generating..."):
                        try:
                            res = generate_answer(question, hits, cfg)
                        except Exception as exc:
                            st.error(f"generation failed: {exc}")
                            continue
                    repeats = render_answer(
                        res["text"], compactor.embedder, ANSWER_THRESHOLD, hits
                    )
                    st.caption(
                        f"{res['prompt_tokens']} prompt + {res['completion_tokens']} "
                        f"completion tokens · {repeats} highlighted statement(s)"
                    )

# ---------------------------------------------------------------------------
# tab 2 - merge inspector
# ---------------------------------------------------------------------------

with tab_merges:
    st.subheader("Every merge, and exactly what it replaced")
    st.caption(
        "The archive keeps each original chunk with its embedding vector, so any "
        "merge can be read back in full and reversed."
    )
    if not merges:
        st.info("No merges yet.")
    else:
        labels = {
            f"{m.merge_id} · {len(m.original_ids)} chunks · "
            f"{' '.join(m.merged_text.split())[:60]}...": m.merge_id
            for m in merges
        }
        picked = st.selectbox("Merge", list(labels))
        merge_id = labels[picked]
        record = compactor.archive.get_merge(merge_id)
        originals = compactor.archive.originals_for(merge_id)

        before_tok = sum(count_tokens(c.text, cfg.token_model) for c, _ in originals)
        after_tok = count_tokens(record.merged_text, cfg.token_model)
        _coverage_preview = merge_coverage(
            [c for c, _v in originals], record.merged_text, compactor.embedder, PROV_THRESHOLD
        )
        _dropped = sum(
            1 for rows in _coverage_preview for _s, state, _sim in rows if state is False
        )
        cols = st.columns(4)
        cols[0].metric("Chunks", f"{len(originals)} → 1")
        cols[1].metric("Tokens", after_tok, delta=after_tok - before_tok)
        cols[2].metric(
            "Reduction",
            f"{(100.0 * (before_tok - after_tok) / before_tok if before_tok else 0):.0f}%",
        )
        cols[3].metric("Sentences dropped", _dropped)

        left, right = st.columns(2)
        coverage = merge_coverage(
            [c for c, _v in originals], record.merged_text, compactor.embedder, PROV_THRESHOLD
        )
        total_dropped = sum(
            1 for rows in coverage for _s, state, _sim in rows if state is False
        )

        with left:
            st.markdown("**Originals (archived)**")
            st.caption(
                "Sentences carried into the merge are tinted in that original's "
                "colour. Red means the sentence did not survive - that is real "
                "information the store no longer holds."
            )
            for i, (chunk, _vec) in enumerate(originals):
                with st.container(border=True):
                    bg, fg = PROV_COLORS[i % len(PROV_COLORS)]
                    st.markdown(
                        f"<span style='background:{bg};color:{fg};padding:2px 9px;"
                        f"border-radius:3px;font-size:11.5px;font-weight:700'>"
                        f"original {i + 1}</span>"
                        f"<span class='rc-meta'> &nbsp; "
                        f"{Path(chunk.source).name if chunk.source else '?'} "
                        f"#{chunk.ordinal} · "
                        f"{count_tokens(chunk.text, cfg.token_model)} tok</span>",
                        unsafe_allow_html=True,
                    )
                    kept, dropped = render_coverage(coverage[i], i)
                    if dropped:
                        st.markdown(
                            f"<div class='rc-meta' style='color:#8E2626;font-weight:600'>"
                            f"{kept} of {kept + dropped} sentences carried over · "
                            f"{dropped} dropped</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div class='rc-meta' style='color:#07514F;font-weight:600'>"
                            f"all {kept} sentences carried over</div>",
                            unsafe_allow_html=True,
                        )
        with right:
            st.markdown("**Merged result (live in the store)**")
            st.caption(
                "Every sentence is tinted by the original it came from. Anything "
                "shown in red matched no original closely - the model wrote it."
            )
            with st.container(border=True):
                st.markdown(
                    f"<div class='rc-meta'>{after_tok} tok · {record.merged_chunk_id}</div>",
                    unsafe_allow_html=True,
                )
                traced = provenance(
                    record.merged_text,
                    [c for c, _v in originals],
                    compactor.embedder,
                    PROV_THRESHOLD,
                )
                untraced = render_provenance(traced)
                st.write("")
                render_prov_legend(len(originals), untraced)
            if untraced:
                st.warning(
                    f"{untraced} sentence(s) could not be traced to any original at "
                    f"the current sensitivity. Lower the slider, or read them closely - "
                    "an untraceable sentence is either a paraphrase or an invention."
                )
            else:
                st.success(
                    "Every sentence in the merged chunk traces back to one of the "
                    "originals. Nothing was invented."
                )
            if st.button("Undo this merge", key=f"undo-{merge_id}"):
                if compactor.undo_merge(merge_id):
                    st.cache_resource.clear()
                    st.success("Originals restored to the store.")
                    st.rerun()
                else:
                    st.warning("Already undone.")

# ---------------------------------------------------------------------------
# tab 3 - embedding map
# ---------------------------------------------------------------------------

with tab_map:
    st.subheader("How the chunks relate")
    st.caption(
        "Each dot is one text chunk. Dots closer together have more similar "
        "content; hover over a dot to read the text. This is a simplified map, "
        "so use it as a visual guide rather than an exact measurement."
    )

    merged_away = set()
    for m in merges:
        merged_away.update(m.original_ids)

    ids = before_store.all_ids()
    if len(ids) < 3:
        st.info("Not enough chunks to project.")
    else:
        matrix = np.vstack([before_store.get_vector(i) for i in ids])
        centre = matrix.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(matrix - centre, full_matrices=False)
        comps = vt[:2]
        coords = (matrix - centre) @ comps.T

        rows = []
        for (x, y), cid in zip(coords, ids):
            chunk = before_store.get(cid)
            rows.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "status": (
                        "Combined into a new chunk"
                        if cid in merged_away
                        else "Kept unchanged"
                    ),
                    "file": source_label(chunk),
                    "text": chunk.text,
                }
            )
        df = pd.DataFrame(rows)

        show_after = st.checkbox(
            "Show chunks after compaction",
            value=False,
            help="Adds the resulting chunks to the same map. Hover over a dot to read its text.",
        )
        chart = (
            alt.Chart(df)
            .mark_circle(size=130, opacity=0.75)
            .encode(
                x=alt.X(
                    "x:Q",
                    title=None,
                    axis=alt.Axis(labels=False, ticks=False, domain=False),
                ),
                y=alt.Y(
                    "y:Q",
                    title=None,
                    axis=alt.Axis(labels=False, ticks=False, domain=False),
                ),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(
                        domain=["Kept unchanged", "Combined into a new chunk"],
                        range=["#9AAAAA", "#A3502E"],
                    ),
                    legend=alt.Legend(title="What happened to the chunk", orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("file:N", title="Source"),
                    alt.Tooltip("status:N", title="Status"),
                    alt.Tooltip("text:N", title="Chunk text"),
                ],
            )
            .properties(height=450)
        )

        if show_after:
            after_ids = after_store.all_ids()
            after_matrix = np.vstack([after_store.get_vector(i) for i in after_ids])
            after_coords = (after_matrix - centre) @ comps.T
            after_rows = [
                {
                    "x": float(x),
                    "y": float(y),
                    "file": source_label(after_store.get(cid)),
                    "text": after_store.get(cid).text,
                }
                for (x, y), cid in zip(after_coords, after_ids)
            ]
            overlay = (
                alt.Chart(pd.DataFrame(after_rows))
                .mark_point(size=90, shape="cross", opacity=0.95, color="#07514F")
                .encode(
                    x=alt.X("x:Q", axis=alt.Axis(labels=False, ticks=False, domain=False)),
                    y=alt.Y("y:Q", axis=alt.Axis(labels=False, ticks=False, domain=False)),
                    tooltip=[
                        alt.Tooltip("file:N", title="Source"),
                        alt.Tooltip("text:N", title="Chunk text"),
                    ],
                )
            )
            chart = chart + overlay

        st.altair_chart(chart, use_container_width=True)
        if show_after:
            st.caption("Teal crosses show the chunks that remain after compaction.")
        c = st.columns(3)
        c[0].metric("Chunks before compaction", len(ids))
        c[1].metric("Chunks combined", len(merged_away))
        c[2].metric("Chunks after compaction", after_store.count())
