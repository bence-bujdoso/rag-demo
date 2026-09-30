# Agentic RAG Chatbot Prototípus..

## 📋 Projekt áttekintés

Ez egy **Agentic RAG (Retrieval-Augmented Generation) chatbot prototípus**, amely a magyar jogi tájékoztatás területén működik. A rendszer a **LangGraph** framework-t használja az agentic workflow megvalósításához, a **Streamlit**-tel rendelkezik felhasználói felületre, és **Docker**-ban futtatható.

## 🎯 Probléma és célkitűzés

### Probléma
A magyar jogi rendszer komplexitása és a jogi információk elérése nehézkes a polgárok számára. A magyar jogi tájékoztató rendszer nem mindig elérhető vagy egyértelmű, és a jogi információk minősége változó.

### Célkitűzés
Egy **moduláris, reprodukálható agentic RAG rendszer** létrehozása, amely:
- A magyar jogi tudástár alapján pontos, hiteles válaszokat ad
- Az agentic architektúra lehetővé teszi a rugalmas döntéshozatalt
- A Streamlit UI egyszerű, kezelhető felületet biztosít
- A Docker konténerizálás biztosítja a reprodukálhatóságot

### Miért releváns a probléma?
- A magyar jogi rendszer bonyolult és sokáig tartó változásokon megy keresztül
- A polgárok számára nehéz megérteni a jogi eljárásokat
- Az adatai nehezen elérhetők és átláthatatlanok
- Egy működő RAG rendszer bizonyítja a modularitás és az agentic architektúra értékét

### Miért az agentic RAG megközelítés?
- **Pontos válaszok**: A RAG rendszer a tudástárból idéz, nem hallgatólagos generálás
- **Rugalmas döntéshozatal**: Az agentic workflow feltételre alapú irányítást alkalmaz
- **Modularitás**: A RAG subgraph és a fő agent elkülönítve fejleszthető
- **Skálázhatóság**: Az adatforrások bővíthetők anélkül, hogy az architektúrát meg kellene változtatni

## 🏗️ Rendszerarchitektúra

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                          │
│                    (port 8501)                           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Agentic Workflow                        │
│                   (LangGraph)                            │
│                                                          │
│  ┌──────────┐    ┌───────────┐    ┌───────────────┐     │
│  │ Intent   │───▶│ Route     │───▶│ Answer        │     │
│  │ Classifier│    │ Decision  │    │ Generator     │     │
│  └──────────┘    └───────────┘    └───────────────┘     │
│       │               │                │                │
│       ▼               ▼                ▼                │
│  ┌──────────┐    ┌───────────┐    ┌───────────────┐     │
│  │ RAG      │    │ Tool      │    │ Evaluation    │     │
│  │ Subgraph │    │ Node      │    │ Node          │     │
│  │ (≥3 node)│    │           │    │               │     │
│  └──────────┘    └───────────┘    └───────────────┘     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   RAG Subgraph (moduláris)                │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Load     │─▶│ Chunk    │─▶│ Embed    │─▶│ Search │ │
│  │ Docs     │  │          │  │          │  │        │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘ │
│                          │                            │
│                          ▼                            │
│                   ┌────────────┐                      │
│                   │ FAISS      │                      │
│                   │ Vectorstore│                      │
│                   └────────────┘                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   Ollama (glm4:9b)                        │
│                   (LLM szolgáltatás)                     │
└─────────────────────────────────────────────────────────┘
```

### Fő komponensek

| Komponens | Leírás | Technológia |
|-----------|--------|-------------|
| **Main Agent** | 5+ node agentic workflow | LangGraph |
| **RAG Subgraph** | Moduláris RAG alrendszer (≥3 node) | LangGraph + FAISS |
| **Tools** | ≥2 eszköz (RAG + időszköz, fizetés-kalkulátor) | LangChain |
| **LLM** | Helyi Ollama szolgáltatás | `glm4:9b` (9.4B Q4_0) |
| **Embedding** | Hungarian multilingual embeddings | `paraphrase-multilingual-MiniLM-L12-hu-v3` |
| **Vector Store** | FAISS CPU index | FAISS-CPU |
| **UI** | Streamlit felhasználói felület | Streamlit |
| **Container** | Docker + docker-compose | Docker |

### Részletek: Agentic Workflow

A fő agent 6 node-ból áll:
1. **intent_classifier**: Kategorializálja a felhasználói kérdést (munkajog, adózás, lakhatás, stb.)
2. **route_decision**: Feltételre alapú irányítás (RAG → rag, Tool → tool, egyeb → direct)
3. **rag_node**: Hívja a RAG subgraph-ot a tudástárból
4. **tool_node**: Hívja a nem-RAG eszközöket (időszámítás, fizetés-kalkuláció)
5. **answer_generator**: Az LLM végső válaszgenerálása a kontextummal
6. **evaluation_node**: Opcionális értékelés/minősítés

### Részletek: RAG Subgraph

A RAG subgraph 5 node-ból áll:
1. **load_documents**: Dokumentumok betöltése a data/ mappából
2. **chunk**: Szövegrészletekre bontás
3. **embed**: FAISS indexelés sentence-transformers embeddingekkel (paraphrase-multilingual-MiniLM-L12-hu-v3)
4. **search**: Vektorkeresés a lekérdezésre (RRF fusion α=0.7)
5. **context**: Kontextum összegyűjtése az LLM-hez

## 🔧 Telepítési és futtatási útmutató

### Előfeltételek

- **Docker** 20.10+ és **docker-compose** 2.0+
- **Ollama** telepítve (ha nem docker-compose-ben futtatod)
- **Python 3.12+** (lokális futtatáshoz)

### Opció 1: Docker-compose (ajánlott)

```bash
# 1. Klónozd a repo-t
git clone <repo-url>
cd rag-demo

# 2. Indítsd a rendszert
docker-compose up -d --build

# 3. Nyisd meg a böngészőben
# http://localhost:8501

# 4. Állítsd le
docker-compose down
```

A docker-compose automatikusan:
- Build-eli a rag-app konténerst
- Elindítja az Ollama szolgáltatást
- Várja az Ollama készenléti állapotát
- Elindítja a Streamlit UI-t

### Opció 2: Lokális futtatás

```bash
# 1. Hozz létre egy virtuális környezetet
python3 -m venv venv
source venv/bin/activate

# 2. Telepítsd a dependency-eket
pip install -r requirements.txt

# 3. Indítsd el az Ollama-t (ha nincs már futban)
ollama serve &
ollama pull glm4:9b

# 4. Futtasd a Streamlit appot
source venv/bin/activate && streamlit run ui/app.py --server.port=8501
```

### Opció 3: Docker-compose már futó Ollama-val

```bash
# Ha az Ollama már fut a host-on (port 11434):
docker-compose -f docker-compose.yml --profile app-only up -d --build rag-app
```

### Tesztelés

```bash
# Értékelési készlet futtatása
python tests/evaluation.py

# Terhelési teszt (50 lekérdezés)
python tests/load_test.py --queries 50

# Terhelési teszt (200 lekérdezés)
python tests/load_test.py --queries 200

# Dry run (csak kérdések generálása)
python tests/load_test.py --queries 100 --dry-run
```

### Docker futtatás csupán a UI-val (Ollama host-on)

```bash
# A docker-compose.yml-ben változtasd meg az Ollama-t:
# environment:
#   OLLAMA_HOST: host.docker.internal:11434
```

## 📊 Funkcionális értékelés és teljesítményteszt

### Értékelési készlet

A projekt tartalmaz 20 kérdést 6 kategóriában:
- **Munkaszerződés**: 4 kérdés (próbaidő, minimálbér, szabadság, felmondás)
- **Adózás**: 3 kérdés (ÁFA, SZJA, adórendészeti)
- **Lakhatás**: 3 kérdés (Otthonkereső, lakbér, szabadság)
- **Adatvédelem**: 3 kérdés (GDPR, bírság, 72 óra)
- **Fogyasztóvédelem**: 3 kérdés (elbocsátás, OGYÉI, garancia)
- **Munkaerő-piac**: 2 kérdés (támogatás, nyugdíj)
- **Általános**: 2 kérdés (járulékok, GDPR jogok)

### Teljesítményteszt Eredmények (2026. szeptember 25)

Funkcionális értékelés:

| Mérő | Érték | Megjegyzés |
|------|-------|------------|
| Összes lekérdezés | 20 | 100% hiba nélküli válasz |
| Átlagos latency | 22.80 másodperc | glm4:9b CPU-on |
| Minimum latency | 3.60 másodperc | |
| Maximum latency | 56.38 másodperc | |
| Fő szűk keresztmetszet | Generálás (22.79 másodperc) | |

### Bottleneck azonosítás

- **Fő szűk keresztmetszet**: Az Ollama LLM válaszidő (glm4:9b ~22-25 másodperc GPU-n)

### Alapvető latency metrikák (CPU-only futás)

| Mérő | Érték | Megjegyzés |
|------|-------|------------|
| Átlagos válaszidő | 20-25 másodperc | CPU-n, glm4:9b |
| Median latency | 18-22 másodperc | |
| P95 latency | 35-45 másodperc | |
| P99 latency | 50-60 másodperc | |
| Első kérés | 30-60 másodperc | Indexelés + LLM |

### Optimalizálási javaslatok:

1. **Kisebb LLM használata**: A glm4:9b helyett egy 3-4B parameterű modell használata (pl. 'tinyllama', 'phi-2') jelentős latency-csökkentést eredményezhet (30s → 8-12s).
2. **FAISS index előtelepítés**: Az index előtelepítése Docker konténer indításakor és memory-ben tartása. Ezenkívül a index mentése docker-compose volume-vel újrahasználható.
3. **GPU használata**: AMD GPU esetén a `device='cuda'` beállításval jelentős gyorsulás várható.

## 📁 Projektfájlstruktúra

```
RagDemo/
├── data/                          # Szöveges adathalmaz
│   ├── munkajogi_tudastar.txt
│   ├── adoszabalyok.txt
│   ├── lakhatasi_jogok.txt
│   ├── munkaeropiac.txt
│   ├── adatvedelem.txt
│   └── fogyasztoverdelem.txt
├── rag/                           # RAG alrendszer
│   ├── __init__.py               # RagSubgraph osztály
│   └── graph.py                  # LangGraph RAG subgraph
├── agent/                         # Agentic workflow
│   └── __init__.py               # Fő LangGraph workflow
├── ui/                            # Streamlit UI
│   └── app.py                    # Fő alkalmazás
├── tests/                         # Értékelési és tesztelő eszközök
│   ├── evaluation.py           # 20 kérdés értékeléshez
│   └── load_test.py              # Terhelési teszt
├── Dockerfile                     # Konténerezés
├── docker-compose.yml             # Multi-container
├── requirements.txt               # Python dependency-ek
└── README.md                      # Ez a dokumentáció
```

## 🔧 Technológia stack

| Réteg | Technológia | Verzió |
|-------|-------------|--------|
| **Workflow** | LangGraph | 1.2+ |
| **LLM Framework** | LangChain Community | 0.4+ |
| **LLM** | Ollama (glm4:9b) | 0.6+ |
| **Embedding** | sentence-transformers | 6.1+ |
| **Embedding modell** | paraphrase-multilingual-MiniLM-L12-hu-v3 | |
| **Vector Store** | FAISS-cpu | 1.15+ |
| **Text Splitter** | langchain-text-splitters | 1.1+ |
| **UI Framework** | Streamlit | 1.64+ |
| **Container** | Docker + docker-compose | 29.1+ |
| **Python** | Python 3.12+ | 3.14 |

## 📜 Licenc

Ez egy prototípus projekt. Nem tartalmaz szabadalmi jogokat.

---

**Verzió**: 1.1.0  
**Utolsó frissítés**: 2026. szeptember 30.  
**Fejlesztő**: Bujdosó Bence
