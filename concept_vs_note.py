# pip install rank_bm25 nltk
from rank_bm25 import BM25Okapi
import nltk, re
from typing import List, Dict, Any
import json
from rare_50 import rare_50
from tqdm import tqdm
# --- one-time (if you haven't already) ---
nltk.download("punkt")
import random

def simple_tokenize(text: str) -> List[str]:
    """Lowercase + basic alnum tokenization."""
    return re.findall(r"[a-z0-9]+", text.lower())

def split_into_sentence_segments(note: str, sentences_per_seg: int = 3, stride: int = 2) -> List[str]:
    """
    Split a long note into overlapping segments of 'sentences_per_seg' sentences,
    moving forward by 'stride' sentences each step.
    """
    sents = [s.strip() for s in nltk.sent_tokenize(note) if s.strip()]
    if not sents:
        return []
    segs = []
    i = 0
    while i < len(sents):
        seg = " ".join(sents[i:i + sentences_per_seg])
        if seg:
            segs.append(seg)
        if i + sentences_per_seg >= len(sents):
            break
        i += stride
    return segs

def bm25_concept_vs_segments(
    note_text: str,
    concepts: List[str],
    sentences_per_seg: int = 3,
    stride: int = 2,
    match_threshold: float = 0.0  # BM25>0 means at least one term overlap
) -> Dict[str, Any]:
    """
    Build BM25 over sentence segments and score each concept vs each segment.
    Returns:
      - segments: list of raw text segments
      - scores:   list of lists [concept_idx][segment_idx] -> float
      - concept_matches: list[bool], concept has any segment with score >= threshold
      - concept_best: list of dicts with best score & segment index
      - matched_count: int number of concepts with at least one match
    """
    # 1) segment
    segments = split_into_sentence_segments(note_text, sentences_per_seg, stride)
    if not segments:
        return {
            "segments": [],
            "scores": [],
            "concept_matches": [False]*len(concepts),
            "concept_best": [{"best_score": 0.0, "best_seg_idx": None} for _ in concepts],
            "matched_count": 0
        }

    # 2) tokenize segments + build BM25
    tokenized_segments = [simple_tokenize(seg) for seg in segments]
    bm25 = BM25Okapi(tokenized_segments)

    # 3) score each concept vs all segments
    scores = []
    concept_matches = []
    concept_best = []

    for concept in concepts:
        q = simple_tokenize(concept)
        seg_scores = bm25.get_scores(q)  # numpy array of floats, len = n_segments
        scores.append(seg_scores.tolist())

        # best segment for this concept
        best_idx = int(max(range(len(seg_scores)), key=lambda i: seg_scores[i])) if len(seg_scores) else None
        best_score = float(seg_scores[best_idx]) if best_idx is not None else 0.0

        has_match = bool(best_score >= match_threshold)
        concept_matches.append(has_match)
        concept_best.append({"best_score": best_score, "best_seg_idx": best_idx})

    matched_count = sum(concept_matches)
    return {
        "segments": segments,
        "scores": scores,
        "concept_matches": concept_matches,
        "concept_best": concept_best,
        "matched_count": matched_count
    }

# ---------- Example ----------

concept_file_1 = "./concepts_layer1_filtered.json" #layer1 or layer2
concept_file_2 = "./concepts_layer2_filtered.json"
data_file = "mimic3-50l" #mimic3-50 or mimic3-50l
type_file = "train" #train, test or dev
concept_picking_threshold = 0.045 # 0.03 for mimic3-50l layer2, 0.045 for mimic3-50l
model_id = "meta-llama/Meta-Llama-3-8B"
model_saving_path = "./rare_model_concept_1/" # model saving path
model_output_path = "./rare_model_concept_1_output/" # model output path
concept_type = "concept_1"
code_list = rare_50[:]


with open(concept_file_1, "r") as f:
        codes = json.load(f)

# with open('../concepts_layer2_deduplicated.json', "r") as f:
#       codes = json.load(f)

concept_dict = {}

for obj in codes:
    if obj['code'] in code_list:
        concept_dict[obj['code']] = obj['concept_1'].split(";")

with open(concept_file_2, "r") as f:
        codes = json.load(f)


for obj in codes:
    if obj['code'] in code_list:
        if obj['code'] not in concept_dict.keys():
            concept_dict[obj['code']] = obj['concept_2'].split(";")
        else:
            concept_dict[obj['code']].extend(obj['concept_2'].split(";"))


if __name__ == "__main__":
    
    with open(f'./{data_file}_{type_file}.json') as f:
        data = json.load(f)
    
    total_labels = 0
    missed_labels_by_concepts = 0
    for d in data:
        medical_note = d["text"]
        p_labels = [l for l in d['labels'].split(';') if l in code_list]
        labels = random.sample([l for l in code_list if l not in p_labels], 2)

        total_labels += len(set(labels))
        
        for l in labels:
            concepts = concept_dict.get(l, [])

            out = bm25_concept_vs_segments(
                medical_note,
                concepts,
                sentences_per_seg=3,  # segment size (sentences)
                stride=2,             # overlap control
                match_threshold=1.0   # >=0 means any overlap; try 0.5–1.5 to be stricter
            )

            #print(f"# segments: {len(out['segments'])}")
            if out['matched_count'] <= 2:
                missed_labels_by_concepts += 1
            #print(f"# concepts with at least one match: {out['matched_count']} / {len(concepts)}\n")
    """
    for i, c in enumerate(concepts):
        b = out["concept_best"][i]
        print(f"- {c!r}: best_score={b['best_score']:.3f}, best_seg_idx={b['best_seg_idx']}")
        if b["best_seg_idx"] is not None:
            print("  segment:", out["segments"][b["best_seg_idx"]][:160], "...")
    """
    print(f'percentage coverage : {(total_labels - missed_labels_by_concepts)/total_labels}')

