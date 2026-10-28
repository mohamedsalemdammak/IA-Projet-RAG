import re
import sys
import unicodedata
from collections import Counter, defaultdict
from math import log

import numpy as np
from sentence_transformers import CrossEncoder

from config import (
    CANDIDATE_K,
    CHUNKS_FILE,
    FAISS_FILE,
    MAX_PER_SOURCE,
    RERANKER_MODEL_NAME,
    RERANK_K,
    TOP_K,
)
from src.embeddings import E5Embedder
from src.vector_store import load_store


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def normalize_text(text):
    text = unicodedata.normalize('NFKC', str(text or ''))
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text).replace('ـ', '')
    for a, b in [('أ','ا'),('إ','ا'),('آ','ا'),('ى','ي'),('ؤ','و'),('ئ','ي')]: text = text.replace(a,b)
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', text.lower())).strip()


def normalize_arabic_words(text):
    replacements = {'انشطة':'نشاط','نشاطات':'نشاط','انشطه':'نشاط','مشاريع':'مشروع','مشروعات':'مشروع','اجتماعات':'اجتماع','تدريبات':'تدريب','دورات':'دورة','فعاليات':'فعالية','برامج':'برنامج','اهداف':'هدف','انجازات':'انجاز','قامت':'قام'}
    words=[]
    for word in normalize_text(text).split():
        if word.startswith('ال') and len(word)>3: word=word[2:]
        words.append(replacements.get(word, word))
    return ' '.join(words)


def get_metadata(chunk): return chunk.get('metadata', {})
def get_source(chunk): return get_metadata(chunk).get('source', '')
def get_filename(chunk): return get_metadata(chunk).get('filename', '')
def get_text(chunk): return chunk.get('text', '')
def extract_year(query):
    years=re.findall(r'\b(20\d{2})\b', str(query)); return years[0] if years else None


def detect_intent(query):
    n=normalize_arabic_words(query)
    activity=['نشاط','فعالية','انشطة','action','activite','activites','activity','activities','evenement','evenements','manifestation']
    project=['مشروع','projet','projets','project','projects','programme','programmes','initiative','initiatives']
    meeting=['اجتماع','reunion','reunions','meeting','meetings','جلسة']
    stats=['احصائيات','احصائية','statistique','statistiques','nombre','combien','عدد','count']
    a=any(x in n for x in activity); p=any(x in n for x in project); m=any(x in n for x in meeting); s=any(x in n for x in stats)
    if a and p:return 'project_activity'
    if p:return 'project'
    if a:return 'activity'
    if s:return 'statistics'
    if m:return 'meeting'
    return 'general'

CONCEPT_MAP={'نشاط':['activité','activités','action','actions','activity'],'فعالية':['activité','événement','event','action'],'مشروع':['projet','projets','project','initiative'],'تدريب':['formation','formations','training'],'اجتماع':['réunion','réunions','meeting'],'activite':['activité','action','actions'],'activites':['activité','action','actions'],'projet':['projet','projets'],'projets':['projet','projects']}

def expand_query(query):
    n=normalize_arabic_words(query); expanded=[n]
    for word in n.split(): expanded.extend(CONCEPT_MAP.get(word, []))
    return ' '.join(expanded)


def build_search_query(user_prompt, conversation_history=None, keywords=None, summary=None):
    """Build a bounded query using the current question and conversation context."""
    sections = [f"Question utilisateur: {str(user_prompt or '').strip()}"]

    if summary:
        sections.append(f"Résumé de la conversation: {str(summary).strip()}")

    if keywords:
        if isinstance(keywords, str):
            keyword_text = keywords.strip()
        else:
            keyword_text = ', '.join(
                str(keyword).strip()
                for keyword in keywords
                if keyword is not None and str(keyword).strip()
            )
        if keyword_text:
            sections.append(f"Mots-clés: {keyword_text}")

    if conversation_history:
        messages = conversation_history[-5:]
        history_lines = []
        for message in messages:
            if isinstance(message, dict):
                role = str(message.get('role', 'message')).strip() or 'message'
                content = str(
                    message.get('content', message.get('text', ''))
                ).strip()
                if content:
                    history_lines.append(f"{role}: {content}")
            else:
                content = str(message).strip()
                if content:
                    history_lines.append(content)
        if history_lines:
            sections.append("Derniers messages:\n" + "\n".join(history_lines))

    return "\n".join(sections)


def content_score(text,intent):
    n=normalize_arabic_words(text); score=0.0
    if intent in ['activity','project','project_activity']:
        strong=['date','date de realisation','lieu','objectifs','objectif','description','directeur d action','partenaire','participants','realisation','realise','action realisee','projet realise','تاريخ','مكان','اهداف','هدف','وصف','شركاء','المشاركين','تم تنفيذ','تنفيذ']
        score += sum(0.12 for x in strong if x in n)
        score += sum(0.04 for x in ['action','actions','activite','activites','evenement','projet','projets','initiative','نشاط','فعالية','مشروع'] if x in n)
    if intent=='meeting': score += sum(0.15 for x in ['reunion','reunions','meeting','assemblee','اجتماع','جلسة'] if x in n)
    if intent=='statistics': score += sum(0.15 for x in ['statistique','statistiques','nombre','count','احصائيات','عدد'] if x in n)
    return min(score,0.60)


def statistics_penalty(text):
    n=normalize_arabic_words(text); patterns=[r'\b\d+\s+actions?\b',r'\b\d+\s+reunions?\b',r'\b\d+\s+formations?\b','statistiques du mois','bilan du mois','monthly statistics','احصائيات الشهر','حصيلة الشهر']; matches=sum(bool(re.search(p,n)) for p in patterns); return -0.35 if matches==1 else (-0.55 if matches>=2 else 0.0)


def administrative_penalty(text,intent):
    if intent not in ['activity','project','project_activity']: return 0.0
    n=normalize_arabic_words(text); words=['bureau executif','bureau','ordre du jour','assemblee generale','agp','congres','president','vice president','secretaire general','tresorier','المكتب التنفيذي','المكتب','جدول الاعمال','الجلسة العامة','المؤتمر']; m=sum(x in n for x in words); return (-0.25 if m>=1 else 0)+(-0.20 if m>=2 else 0)


def cover_penalty(text):
    n=normalize_arabic_words(text); words=n.split(); p=-0.35 if len(words)<=12 else 0.0
    patterns=['rapport janvier','rapport fevrier','rapport mars','rapport avril','rapport mai','rapport juin','rapport juillet','rapport aout','rapport septembre','rapport octobre','rapport novembre','rapport decembre','rapport moral','rapport financier','plan d action 2024','sidi mansour','olm sidi mansour']; m=sum(x in n for x in patterns)
    if m>=1 and len(words)<35:p-=0.25
    if m>=2:p-=0.15
    return p


def year_signals(text, source, requested_year):
    """Return year signals while distinguishing the document year from mentions inside it."""
    if not requested_year:
        return 0.0, 0.0, 0.0

    source_years=re.findall(r'\b20\d{2}\b', normalize_text(source))
    text_years=re.findall(r'\b20\d{2}\b', normalize_text(text))
    source_has=bool(source_years)
    source_exact=requested_year in source_years
    source_other=bool([y for y in source_years if y != requested_year])
    text_exact=requested_year in text_years

    # Strong signal: the document itself is explicitly labelled with the requested year.
    source_score = 4.0 if source_exact else (-2.5 if source_other else 0.0)

    # A chunk may mention the requested year historically (e.g. a 2025 report describing 2024).
    # Keep it eligible, but weaker than a document whose filename/source is from that year.
    text_score = 1.5 if text_exact else 0.0

    # Backward-compatible overall year score used by the filtering stage.
    if source_exact:
        overall = 5.0
    elif text_exact and not source_other:
        overall = 3.0
    elif text_exact and source_other:
        overall = 1.0
    elif source_other:
        overall = -5.0
    else:
        overall = -1.0
    return overall, source_score, text_score


class BM25Index:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.document_lengths = []
        self.postings = defaultdict(list)

        for index, document in enumerate(documents):
            frequencies = Counter(normalize_text(document).split())
            length = sum(frequencies.values())
            self.document_lengths.append(length)
            for term, frequency in frequencies.items():
                self.postings[term].append((index, frequency))

        self.document_count = len(documents)
        self.average_length = (
            sum(self.document_lengths) / self.document_count
            if self.document_count
            else 0.0
        )

    def score(self, query):
        scores = [0.0] * self.document_count
        if not self.average_length:
            return scores

        for term in set(normalize_text(query).split()):
            documents = self.postings.get(term, ())
            if not documents:
                continue
            document_frequency = len(documents)
            inverse_document_frequency = log(
                1 + (self.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for index, frequency in documents:
                length = self.document_lengths[index]
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self.average_length
                )
                scores[index] += (
                    inverse_document_frequency
                    * frequency
                    * (self.k1 + 1)
                    / denominator
                )
        return scores


def apply_year_filter(candidates, requested_year):
    if not requested_year:
        return candidates
    exact_source=[x for x in candidates if x['source_year_score'] >= 4]
    historical=[x for x in candidates if x['source_year_score'] < 4 and x['year_score'] >= 1]
    no_year=[x for x in candidates if x['year_score'] == -1]
    other=[x for x in candidates if x['year_score'] < -1]

    # Prefer documents explicitly belonging to the requested year. Historical mentions
    # remain available as supporting evidence, but cannot outrank exact-year documents.
    if exact_source:
        return exact_source + historical + no_year
    return historical + no_year + other


def diversify(results):
    final=[]; counts={}
    for item in results:
        source=item['source']; counts[source]=counts.get(source,0)
        if counts[source]>=MAX_PER_SOURCE:continue
        final.append(item); counts[source]+=1
        if len(final)>=TOP_K:break
    return final


class HybridRetriever:
    def __init__(self,index,chunks):
        self.index=index; self.chunks=chunks
        self.bm25_index=BM25Index([get_text(chunk) for chunk in chunks])
        self.embedding_model=E5Embedder()
        self.reranker=CrossEncoder(RERANKER_MODEL_NAME)

    def search(
        self,
        query,
        conversation_history=None,
        keywords=None,
        summary=None,
        prepared_query=None,
    ):
        search_query = prepared_query or build_search_query(
            query,
            conversation_history=conversation_history,
            keywords=keywords,
            summary=summary,
        )
        requested_year=extract_year(search_query); intent=detect_intent(search_query); expanded=expand_query(search_query)
        q=np.asarray(self.embedding_model.embed_query(expanded),dtype='float32').reshape(1,-1)
        scores,indices=self.index.search(q,CANDIDATE_K)
        candidates_by_index={}
        for rank,idx in enumerate(indices[0]):
            if idx<0 or idx>=len(self.chunks):continue
            candidates_by_index[int(idx)] = float(scores[0][rank])

        bm25_scores=self.bm25_index.score(search_query)
        lexical_indices=sorted(
            (idx for idx,score in enumerate(bm25_scores) if score>0),
            key=lambda idx:bm25_scores[idx],
            reverse=True,
        )[:CANDIDATE_K]
        max_bm25_score=max((bm25_scores[idx] for idx in lexical_indices),default=0.0)
        for idx in lexical_indices:
            candidates_by_index.setdefault(idx,0.0)

        candidates=[]
        for idx,semantic in candidates_by_index.items():
            chunk=self.chunks[idx]; text=get_text(chunk); source=get_source(chunk)
            bm25_score=bm25_scores[idx]
            lexical_score=bm25_score/max_bm25_score if max_bm25_score else 0.0
            ys, source_ys, text_ys = year_signals(text,source,requested_year); cs=content_score(text,intent); sp=statistics_penalty(text); ap=administrative_penalty(text,intent); cp=cover_penalty(text)
            candidates.append({'index':idx,'text':text,'source':source,'filename':get_filename(chunk),'semantic_score':semantic,'lexical_score':lexical_score,'bm25_score':bm25_score,'year_score':ys,'source_year_score':source_ys,'text_year_score':text_ys,'content_score':cs,'statistics_penalty':sp,'admin_penalty':ap,'cover_penalty':cp,'preliminary_score':semantic*.30+lexical_score*.30+ys+source_ys+text_ys+cs+sp+ap+cp})
        candidates=apply_year_filter(candidates,requested_year); candidates.sort(key=lambda x:x['preliminary_score'],reverse=True); rerank_candidates=candidates[:RERANK_K]
        if rerank_candidates:
            rr=self.reranker.predict([[search_query,x['text']] for x in rerank_candidates],show_progress_bar=False)
        else: rr=[]
        for item,score in zip(rerank_candidates,rr):
            item['reranker_score']=float(score); item['final_score']=float(score)+item['year_score']+item['source_year_score']+item['text_year_score']+item['content_score']+item['statistics_penalty']+item['admin_penalty']+item['cover_penalty']+item['semantic_score']*.10+item['lexical_score']*.10
        results=diversify(sorted(rerank_candidates,key=lambda x:x['final_score'],reverse=True))
        return {'query':query,'search_query':search_query,'intent':intent,'requested_year':requested_year,'expanded_query':expanded,'search_methods':['semantic','bm25'],'results':results}


def main():
    prompts = [
        "Quels sont les projets réalisés en 2024 ?",
        "ما هي أهم الأنشطة والفعاليات المنجزة في سنة 2025؟",
        "Quelles réunions ont été organisées en 2025 ?",
        "Donne les statistiques des activités.",
    ]

    index, chunks = load_store(FAISS_FILE, CHUNKS_FILE)
    retriever = HybridRetriever(index, chunks)

    for prompt in prompts:
        result = retriever.search(prompt)
        print(f"\n{'=' * 80}\nPrompt: {prompt}")
        print(
            f"Intent: {result['intent']} | "
            f"Année: {result['requested_year'] or 'non spécifiée'}"
        )
        if not result["results"]:
            print("Aucun résultat.")
            continue

        for rank, item in enumerate(result["results"], start=1):
            print(
                f"\n#{rank} score={item['final_score']:.4f} | "
                f"source={item['source']}"
            )
            print(item["text"][:500].replace("\n", " "))


if __name__ == "__main__":
    main()
