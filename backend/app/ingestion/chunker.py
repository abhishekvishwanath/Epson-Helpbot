"""Structure-based hierarchical chunking with token-based fallback.

Hierarchy (as in the reference architecture):

* **Level 1 – parent section** (chapter), **Level 2 – subsection** (``##``),
  **Level 3 – child chunk** cut from a subsection.
* A ``##`` subsection is kept whole when it is within ``h3_word_threshold``
  words (the "H3 gate"); above it, it is split at its ``###`` children.
* ``###`` nodes split further at their own sub-headings only when they exceed
  the target size; deeper nodes (e.g. a problem with its causes and solutions)
  stay atomic up to the hard maximum.
* Anything still too large falls back to the smallest meaningful boundary:
  procedure group → block/paragraph → sentence. Tables split by row groups
  with the header repeated. Warnings, figures and cross-references stay
  attached to the instruction they follow.

Every chunk carries its full heading path as a context header.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingestion.document_builder import render_block
from app.ingestion.metadata import Chunk, classify_chunk, content_hash, stable_id
from app.ingestion.models import Block, BlockType, ParsedDocument, SectionNode
from app.ingestion.tokens import TokenCounter

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(•])")
_XREF_RE = re.compile(r"“([^”]+)” on page (\d+)")
_ATTACH_TO_PREVIOUS = {BlockType.CALLOUT, BlockType.FIGURE, BlockType.LINK}
_MIN_CHUNK_TOKENS = 80


@dataclass
class ChunkingConfig:
    target_tokens: int = 550
    max_tokens: int = 900
    overlap_tokens: int = 60
    h3_word_threshold: int = 400


@dataclass
class _Unit:
    text: str
    tokens: int
    blocks: list[Block]
    group: int | None = None  # units sharing a group (procedure/table) prefer staying together
    sticky_next: bool = False  # labels such as "Solutions" bind to what follows
    table_rows: bool = False  # row group of a split table (self-describing via its header)


@dataclass
class _Draft:
    node: SectionNode
    units: list[_Unit]
    fallback: bool = False
    continued: bool = False
    overlap: str = ""
    parts: list[str] = field(default_factory=list)


class HierarchicalChunker:
    def __init__(self, counter: TokenCounter, config: ChunkingConfig | None = None) -> None:
        self.counter = counter
        self.cfg = config or ChunkingConfig()
        self._group_seq = 0

    # ------------------------------------------------------------------ public
    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        self._doc = doc
        self._paths: dict[str, list[SectionNode]] = {}
        self._index_paths(doc.root, [])
        drafts: list[_Draft] = []
        if doc.root.blocks:
            front = self._chunk_own_blocks(doc.root, doc.root.blocks)
            # Cover-page fragments carry no answerable content.
            drafts += [d for d in front if sum(u.tokens for u in d.units) >= 40]
        for child in doc.root.children:
            drafts += self._chunk_node(child)
        return self._finalise(drafts)

    # ------------------------------------------------------------- tree walk
    def _index_paths(self, node: SectionNode, path: list[SectionNode]) -> None:
        self._paths[node.section_id] = path
        for child in node.children:
            self._index_paths(child, path + [child])

    def _subtree_blocks(self, node: SectionNode) -> list[tuple[SectionNode, Block]]:
        return [(n, b) for n in node.walk() for b in n.blocks]

    def _words(self, node: SectionNode) -> int:
        return sum(len(b.text.split()) for _, b in self._subtree_blocks(node))

    def _tokens(self, node: SectionNode) -> int:
        return self.counter.count(self._render_subtree(node))

    def _body_budget(self, node: SectionNode) -> int:
        return self.cfg.max_tokens - self.counter.count(self._header(node)) - 4

    def _chunk_node(self, node: SectionNode) -> list[_Draft]:
        tokens = self._tokens(node)
        if node.children:
            if node.level <= 1:
                split = True  # chapters are always split into their subsections
            elif node.level == 2:
                split = self._words(node) > self.cfg.h3_word_threshold or tokens > self._body_budget(node)
            elif node.level == 3:
                split = tokens > self.cfg.target_tokens
            else:
                split = tokens > self._body_budget(node)
            if split:
                return self._split_at_children(node)
        if tokens <= self._body_budget(node):
            return [_Draft(node=node, units=[self._whole_unit(node)])]
        return self._chunk_own_blocks(node, node.blocks)  # leaf too large → fallback

    def _split_at_children(self, node: SectionNode) -> list[_Draft]:
        intro = self._chunk_own_blocks(node, node.blocks) if node.blocks else []
        children: list[_Draft] = []
        for child in node.children:
            children += self._chunk_node(child)
        # A short lead-in ("One of the following may be the cause.") is context,
        # not a chunk of its own: fold it into the first child.
        if len(intro) == 1 and children and sum(u.tokens for u in intro[0].units) < _MIN_CHUNK_TOKENS:
            children[0].units = intro[0].units + children[0].units
            return children
        return intro + children

    # ------------------------------------------------------------ rendering
    def _render_subtree(self, node: SectionNode, depth: int = 0) -> str:
        parts = [] if depth == 0 else [f"{'#' * min(depth + 2, 6)} {node.title}"]
        parts += [t for t in (render_block(b) for b in node.blocks) if t]
        parts += [self._render_subtree(c, depth + 1) for c in node.children]
        return "\n\n".join(p for p in parts if p)

    def _whole_unit(self, node: SectionNode) -> _Unit:
        text = self._render_subtree(node)
        return _Unit(text=text, tokens=self.counter.count(text), blocks=[b for _, b in self._subtree_blocks(node)])

    def _header(self, node: SectionNode) -> str:
        titles = [n.title for n in self._paths.get(node.section_id, [])]
        return "[" + " > ".join([self._doc.document_name, *titles]) + "]"

    # ------------------------------------------------------------- fallback
    def _units_from_blocks(self, blocks: list[Block], budget: int) -> list[_Unit]:
        units: list[_Unit] = []
        procedure: int | None = None
        for block in blocks:
            text = render_block(block)
            if not text:
                continue
            tokens = self.counter.count(text)
            if block.type in _ATTACH_TO_PREVIOUS and units and units[-1].tokens + tokens <= budget:
                prev = units[-1]
                prev.text += "\n\n" + text
                prev.tokens += tokens
                prev.blocks.append(block)
                continue
            if units and units[-1].sticky_next and units[-1].tokens + tokens <= budget:
                prev = units[-1]
                prev.text += "\n\n" + text
                prev.tokens += tokens
                prev.blocks.append(block)
                prev.sticky_next = False
                continue
            if block.type == BlockType.STEP:
                if procedure is None or block.text.startswith("1."):
                    self._group_seq += 1
                    procedure = self._group_seq
                group = procedure
            else:
                if block.type not in (BlockType.BULLET,):
                    procedure = None
                group = procedure
            if tokens > budget:
                units += self._split_oversized(block, text, budget)
                continue
            units.append(
                _Unit(text=text, tokens=tokens, blocks=[block], group=group, sticky_next=block.type == BlockType.LABEL)
            )
        return units

    def _split_oversized(self, block: Block, text: str, budget: int) -> list[_Unit]:
        self._group_seq += 1
        group = self._group_seq
        if block.type == BlockType.TABLE and block.table is not None:
            table = block.table
            units, rows = [], []
            for row in table.rows:
                candidate = table.to_markdown(rows + [row])
                if rows and self.counter.count(candidate) > budget:
                    md = table.to_markdown(rows)
                    units.append(_Unit(text=md, tokens=self.counter.count(md), blocks=[block], group=group, table_rows=True))
                    rows = []
                rows.append(row)
            if rows:
                md = table.to_markdown(rows)
                units.append(_Unit(text=md, tokens=self.counter.count(md), blocks=[block], group=group, table_rows=True))
            return units
        # Sentence-level fallback for an over-long paragraph/step.
        units, current = [], ""
        for sentence in _SENTENCE_RE.split(text):
            candidate = f"{current} {sentence}".strip()
            if current and self.counter.count(candidate) > budget:
                units.append(_Unit(text=current, tokens=self.counter.count(current), blocks=[block], group=group))
                current = sentence
            else:
                current = candidate
        if current:
            units.append(_Unit(text=current, tokens=self.counter.count(current), blocks=[block], group=group))
        return units

    def _chunk_own_blocks(self, node: SectionNode, blocks: list[Block]) -> list[_Draft]:
        budget = self._body_budget(node)
        units = self._units_from_blocks(blocks, budget)
        if not units:
            return []
        total = sum(u.tokens for u in units)
        if total <= budget:
            return [_Draft(node=node, units=units)]

        packs: list[list[_Unit]] = []
        current: list[_Unit] = []
        for unit in units:
            size = sum(u.tokens for u in current)
            same_group = current and unit.group is not None and unit.group == current[-1].group
            if not current or size + unit.tokens <= self.cfg.target_tokens:
                current.append(unit)
            elif size + unit.tokens <= budget and (same_group or size < _MIN_CHUNK_TOKENS):
                current.append(unit)
            else:
                packs.append(current)
                current = [unit]
        if current:
            if packs and sum(u.tokens for u in current) < _MIN_CHUNK_TOKENS and sum(
                u.tokens for u in packs[-1] + current
            ) <= budget:
                packs[-1] += current
            else:
                packs.append(current)

        drafts = []
        for i, pack in enumerate(packs):
            draft = _Draft(node=node, units=pack, fallback=True)
            if i and pack[0].group is not None and pack[0].group == packs[i - 1][-1].group and not pack[0].table_rows:
                draft.continued = True
                draft.overlap = self.counter.tail(packs[i - 1][-1].text, self.cfg.overlap_tokens)
            drafts.append(draft)
        return drafts

    # ------------------------------------------------------------- finalise
    def _section_ids(self, node: SectionNode) -> tuple[list[str], list[str]]:
        path = self._paths.get(node.section_id, [])
        titles = [n.title for n in path]
        ids = [stable_id(self._doc.document_id, *titles[: i + 1]) for i in range(len(titles))]
        return titles, ids

    def _finalise(self, drafts: list[_Draft]) -> list[Chunk]:
        doc = self._doc
        chunks: list[Chunk] = []
        per_section: dict[str, list[Chunk]] = {}
        seen_ids: dict[str, int] = {}
        for index, draft in enumerate(drafts):
            titles, ids = self._section_ids(draft.node)
            section_key = ids[-1] if ids else stable_id(doc.document_id, "front-matter")
            body_parts = [u.text for u in draft.units]
            if draft.continued:
                body_parts.insert(0, f"(Continued from the previous part of this section) …{draft.overlap}")
            body = "\n\n".join(body_parts).strip()
            header = self._header(draft.node)
            blocks = [b for u in draft.units for b in u.blocks]
            pages = [b.page_index for b in blocks] or [draft.node.page_index]
            first, last = min(pages), max(pages)
            total = max(self.counter.count(body), 1)
            shares: dict[str, float] = {}
            for u in draft.units:
                kind = u.blocks[0].type.value if len(u.blocks) == 1 else "mixed"
                shares[kind] = shares.get(kind, 0.0) + u.tokens / total
            labels = [b.text for b in blocks if b.type == BlockType.LABEL]
            steps = sum(1 for b in blocks if b.type == BlockType.STEP)
            heading_titles = titles + [n.title for n in draft.node.walk()][1:]
            questions = sum(1 for t in heading_titles if t.rstrip().endswith("?"))
            sub_titles = [n.title for n in draft.node.walk()][1:]

            seq = seen_ids.get(section_key, 0) + 1
            seen_ids[section_key] = seq
            chunk = Chunk(
                chunk_id=f"{doc.document_id}::{section_key}::{seq:02d}",
                chunk_index=index,
                document_id=doc.document_id,
                document_name=doc.document_name,
                document_version=doc.source_hash[:12],
                manufacturer=doc.manufacturer,
                product_model=doc.product_model,
                parent_section=titles[0] if titles else "",
                subsection=titles[1] if len(titles) > 1 else "",
                topic=titles[2] if len(titles) > 2 else "",
                heading_path=titles,
                section_title=titles[-1] if titles else doc.document_name,
                parent_id=ids[0] if ids else "",
                subsection_id=ids[1] if len(ids) > 1 else "",
                topic_id=ids[2] if len(ids) > 2 else "",
                section_id=section_key,
                chunk_type=classify_chunk(titles + sub_titles[:3], labels, shares, steps, questions),
                has_procedure=steps >= 2,
                has_warning=any(b.type == BlockType.CALLOUT for b in blocks),
                has_table=any(b.type == BlockType.TABLE for b in blocks),
                pdf_page_start=first + 1,
                pdf_page_end=last + 1,
                page_start=doc.pages[first].printed_label,
                page_end=doc.pages[last].printed_label,
                token_count=self.counter.count(f"{header}\n\n{body}"),
                word_count=len(body.split()),
                fallback_split=draft.fallback,
                context_header=header,
                content=body,
                content_hash=content_hash(body),
                figure_ids=[b.figure_id for b in blocks if b.figure_id],
                cross_references=[f"“{t}” on page {p}" for t, p in _XREF_RE.findall(body)],
            )
            chunks.append(chunk)
            per_section.setdefault(section_key, []).append(chunk)

        for siblings in per_section.values():
            for i, c in enumerate(siblings):
                c.split_part, c.split_total = i + 1, len(siblings)
                c.prev_chunk_id = siblings[i - 1].chunk_id if i else None
                c.next_chunk_id = siblings[i + 1].chunk_id if i + 1 < len(siblings) else None
        return chunks
