from typing import List, Dict, Any


SYSTEM_PROMPT = """You are a rigorous, truthful academic research assistant powered by Reinforcement Learning from Human Feedback.
Your objective is to provide comprehensive, factual, and deeply technical answers based strictly on the retrieved academic evidence.

Strict Rules:
1. Every major technical claim must cite its source using inline bracketed notation: [1], [2], etc.
2. Ground all factual statements in the provided research paper passages. Do NOT extrapolate or invent facts not present in the evidence.
3. Write direct, authoritative academic assertions followed by numerical citations (e.g., 'Model scaling does not guarantee intent alignment [1].'). Do NOT write repetitive prefixes like 'According to Paper [1]' — let the citation numbers link to the references.
4. Structure your response under a '### Research Answer' heading, presenting clear technical paragraphs.
5. End your response with a '### References' section mapping each citation index [k] to the paper title and arXiv ID."""


def format_evidence_block(evidence_list: List[Dict[str, Any]]) -> str:
    if not evidence_list:
        return "No external evidence retrieved. Answer based on foundational scientific principles and state any uncertainty."

    blocks = []
    for doc in evidence_list:
        idx = doc.get("citation_index", len(blocks) + 1)
        title = doc.get("title", "Untitled")
        arxiv_id = doc.get("arxiv_id", "N/A")
        text = doc.get("text", "").strip()
        authors = ", ".join(doc.get("authors", [])[:3])
        if len(doc.get("authors", [])) > 3:
            authors += " et al."
        blocks.append(
            f"[{idx}] Title: {title} (arXiv: {arxiv_id})\n"
            f"    Authors: {authors}\n"
            f"    Evidence Excerpt: {text}"
        )
    return "\n\n".join(blocks)


def build_research_prompt(query: str, evidence_list: List[Dict[str, Any]]) -> str:
    evidence_text = format_evidence_block(evidence_list)
    prompt = f"""### Context & Retrieved Academic Evidence:
{evidence_text}

### Research Query:
{query}

### Instruction:
Synthesize the retrieved evidence to answer the research query with precision, clarity, and inline bracketed citations [1], [2]. Ensure every claim is grounded."""
    return prompt
