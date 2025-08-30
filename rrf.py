from typing import List

def reciprocal_rank_fusion(results_lists: List[List], k: int = 60, top_n: int = 5):
    score_dict = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            doc_id = (doc.page_content, frozenset(doc.metadata.items()))
            score = 1 / (k + rank + 1)
            if doc_id in score_dict:
                score_dict[doc_id]["score"] += score
            else:
                score_dict[doc_id] = {"score": score, "doc": doc}
                
    ranked = sorted(score_dict.values(), key=lambda x: x["score"], reverse=True)
    return [(item["doc"], item["score"]) for item in ranked[:top_n]]
