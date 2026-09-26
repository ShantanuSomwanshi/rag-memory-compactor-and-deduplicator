"""Side-by-side retrieval demo: the same query against the store before and
after compaction.

Run with:  streamlit run app.py

The "before" index is not a saved copy - it is reconstructed from the live store
plus the archive, which is only possible because every merge archived its
originals together with their embedding vectors.
"""

from __future__ import annotations

import itertools
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
      .rc-dup  { background: #F6E3D8; border-left: 3px solid #A3502E;
                 padding: 2px 6px; border-radius: 2px; }
      .rc-ans  { font-size: 15px; line-height: 1.7; }
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


def render_answer(text: str, embedder, threshold: float) -> int:
    sents, repeats = answer_redundancy(text, embedder, threshold)
    if not sents:
        st.write(text)
        return 0
    parts = []
    for idx, sentence in enumerate(sents):
        if idx in repeats:
            src, sim = repeats[idx]
            parts.append(
                f"<span class='rc-dup' title='repeats statement {src + 1} "
                f"(similarity {sim:.2f})'>{sentence}</span>"
            )
        else:
            parts.append(sentence)
    st.markdown(f"<div class='rc-ans'>{' '.join(parts)}</div>", unsafe_allow_html=True)
    return len(repeats)


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
        if len(text) > 400:
            with st.expander("full text"):
                st.write(text)


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

with st.sidebar:
    st.subheader("Index")
    st.metric("Before compaction", f"{before_store.count()} chunks")
    st.metric("After compaction", f"{after_store.count()} chunks")
    st.caption(f"{len(merges)} merges applied")
    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("chunks retrieved (top-k)", 3, 10, 5)
    want_answers = st.checkbox("Generate answers from both contexts", value=True)
    answer_threshold = st.slider(
        "answer-repetition sensitivity", 0.50, 0.95, 0.68, 0.01,
        help="Two sentences in an answer above this similarity are flagged as a repeat.",
    )
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

        if want_answers:
            st.divider()
            st.subheader("Answers generated from each context")
            st.caption(
                "Shaded statements repeat something already said in the same answer — "
                "found with the same embedding comparison the compactor uses on the corpus."
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
                    repeats = render_answer(res["text"], compactor.embedder, answer_threshold)
                    st.caption(
                        f"{res['prompt_tokens']} prompt + {res['completion_tokens']} "
                        f"completion tokens · {repeats} repeated statement(s)"
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
        cols = st.columns(3)
        cols[0].metric("Chunks", f"{len(originals)} → 1")
        cols[1].metric("Tokens", after_tok, delta=after_tok - before_tok)
        cols[2].metric(
            "Reduction",
            f"{(100.0 * (before_tok - after_tok) / before_tok if before_tok else 0):.0f}%",
        )

        left, right = st.columns(2)
        with left:
            st.markdown("**Originals (archived)**")
            for i, (chunk, _vec) in enumerate(originals, 1):
                with st.container(border=True):
                    st.markdown(
                        f"<div class='rc-meta'>{i}/{len(originals)} · "
                        f"{Path(chunk.source).name if chunk.source else '?'} "
                        f"#{chunk.ordinal} · "
                        f"{count_tokens(chunk.text, cfg.token_model)} tok</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='rc-text'>{' '.join(chunk.text.split())}</div>",
                        unsafe_allow_html=True,
                    )
        with right:
            st.markdown("**Merged result (live in the store)**")
            with st.container(border=True):
                st.markdown(
                    f"<div class='rc-meta'>{after_tok} tok · {record.merged_chunk_id}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='rc-text'>{' '.join(record.merged_text.split())}</div>",
                    unsafe_allow_html=True,
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
    st.subheader("Where the duplicates sit in embedding space")
    st.caption(
        "Every chunk projected to two dimensions (principal components of the "
        "pre-compaction vectors). Illustrative, not evidential: two axes cannot "
        "carry 384 dimensions. Duplicate pairs land on top of each other."
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
                    "state": "merged away (duplicate)" if cid in merged_away else "kept as-is",
                    "source": Path(chunk.source).name if chunk.source else "?",
                    "text": " ".join(chunk.text.split())[:150] + "...",
                }
            )
        df = pd.DataFrame(rows)

        show_after = st.checkbox("Overlay the compacted index", value=False)
        chart = (
            alt.Chart(df)
            .mark_circle(size=130, opacity=0.75)
            .encode(
                x=alt.X("x:Q", title="component 1"),
                y=alt.Y("y:Q", title="component 2"),
                color=alt.Color(
                    "state:N",
                    scale=alt.Scale(
                        domain=["kept as-is", "merged away (duplicate)"],
                        range=["#9AAAAA", "#A3502E"],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["source", "state", "text"],
            )
            .properties(height=520)
        )

        if show_after:
            after_ids = after_store.all_ids()
            after_matrix = np.vstack([after_store.get_vector(i) for i in after_ids])
            after_coords = (after_matrix - centre) @ comps.T
            after_rows = [
                {
                    "x": float(x),
                    "y": float(y),
                    "state": "after compaction",
                    "source": source_label(after_store.get(cid)),
                    "text": " ".join(after_store.get(cid).text.split())[:150] + "...",
                }
                for (x, y), cid in zip(after_coords, after_ids)
            ]
            overlay = (
                alt.Chart(pd.DataFrame(after_rows))
                .mark_point(size=90, shape="cross", opacity=0.95, color="#07514F")
                .encode(x="x:Q", y="y:Q", tooltip=["source", "text"])
            )
            chart = chart + overlay

        st.altair_chart(chart, use_container_width=True)
        c = st.columns(3)
        c[0].metric("Chunks before", len(ids))
        c[1].metric("Merged away", len(merged_away))
        c[2].metric("Chunks after", after_store.count())
