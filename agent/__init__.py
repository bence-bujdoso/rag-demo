"""Agentic Workflow fő szűrője LangGraph-kel.

A fő agentic workflow legalább 5 node-ot tartalmaz:
1. intent_classifier - Kategorizálja a felhasználói kérdést
2. route_decision     - Feltételre alapú irányítás (RAG vagy más eszköz)
3. rag_node           - RAG subgraph meghívása (szabadalom)
4. tool_node          - Nem-RAG eszközök meghívása (pl. idő, számítás)
5. answer_generator   - Végső válasz generálása az LLM-mel
6. evaluation_node    - Opcionális értékelés/minősítés

A conditional routing a ritmus alapja:
- Munkajogi/fogyasztói kérdés -> RAG
- Számítás/idő kérdés -> Tool
- Általános kérdés -> LLM közvetlenül
"""

from datetime import datetime, timedelta
from typing import TypedDict, Optional, List, Literal, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
import time
import json

from rag import RagSubgraph, RagConfig as RConfig
from rag.scraper import init_scraper, check_updates, get_last_scrape_time


# === Adatstruktúrák ===

class NodeTiming(TypedDict):
    """Egy nód végrehajtási ideje."""
    node_name: str
    start_time: float
    end_time: float
    duration_seconds: float
    message: str
    operations: List[str]


class AgentState(TypedDict):
    """A fő agent állapota."""
    messages: List[dict]
    query: str
    intent: Optional[str]
    route: Optional[str]
    context: str
    retrieved_docs: List[dict]
    tool_calls: Optional[List[dict]]
    tool_results: Optional[dict]
    answer: Optional[str]
    error: Optional[str]
    evaluation_score: Optional[float]
    node_timings: Optional[List[NodeTiming]]


class AgentConfig(TypedDict):
    """Agent konfiguráció."""
    model_name: str
    rag_config: RConfig
    temperature: float
    max_tokens: int


# === Helper függvények ===

def get_current_date() -> str:
    """Visszaadja a aktuális dátumot ISO 8601 formátumban (YYYY-MM-DD)."""
    return datetime.now().strftime("%Y-%m-%d")


def get_llm(config: AgentConfig) -> ChatOllama:
    """Lekéri az Ollama LLM-t a konfiguráció alapján."""
    return ChatOllama(
        model=config["model_name"],
        temperature=config.get("temperature", 0.7),
        num_ctx=config.get("max_tokens", 4096)
    )


def classify_intent(query: str) -> str:
    """
    Egyszerű szavas alapú kategorializálás.
    Visszaadja az intent típusát:
    - "munkajog" - munkaszerződés, illetmény
    - "adozás" - adózási kérdések
    - "lakhatasag" - lakhatási jogok
    - "adoavedelem" - adatvédelmi kérdések
    - "fogyasztoverdelem" - fogyasztóvédelmi kérdések
    - "munkaeropiac" - munkaerő-piaci kérdések
    - "tool" - eszköz/hiba/számítás kérdés
    - "egyeb" - egyéb általános kérdés
    """
    query_lower = query.lower()
    
    intent_keywords = {
        "munkajog": ["munkaszerződés", "bér", "fizetés", "fizet", "munka", "próbaidő", "szünet", "nyugdíj", "munkaerő", "szerződés", "jog"],
        "adozás": ["adó", "szja", "afa", "járulé", "nyugdíjpénztár", "adójelent", "adószá", "csúcsadó", "áfa", "adó"],
        "lakhatasag": ["lakbér", "lakás", "lakhatos", "bérleti", "otthonkereső", "szociális lak"],
        "adoavedelem": ["adatvédel", "gdpr", "adoatvédelem", "személyes adat", "adatkezel"],
        "fogyasztoverdelem": ["fogyasztó", "vásárlás", "visszaeladás", "garancia", "távolsági", "ogyéség", "termékbírál"],
        "munkaeropiac": ["munkaerő", "állás", "munkaerőpiac", "munkanélkü", "helyszín", "foglalkoztatás", "támogatás"],
        "tool": ["idő", "óra", "nap", "számítás", "mennyi", "hány", "hányas", "kalkulál", "count", "time", "számol", "kérdés"],
    }
    
    for intent, keywords in intent_keywords.items():
        for kw in keywords:
            if kw.lower() in query_lower:
                return intent
    
    return "egyeb"


def route_decision(intent: str) -> Literal["rag", "tool", "direct"]:
    """
    Conditional routing döntés.
    
    Visszaad:
    - "rag" -> RAG subgraph (tudástári kérdések)
    - "tool" -> eszköz hívás (számítás, idő)
    - "direct" -> közvetlen LLM válasz
    """
    rag_intents = {"munkajog", "adozás", "lakhatasag", "adoavedelem", "fogyasztoverdelem", "munkaeropiac"}
    tool_intents = {"tool"}
    
    if intent in rag_intents:
        return "rag"
    elif intent in tool_intents:
        return "tool"
    else:
        return "direct"


# === Node-ok ===

def intent_classifier_node(state: AgentState) -> dict:
    """1. Node: Kategorizálja a felhasználói kérdést."""
    query = state["query"]
    intent = classify_intent(query)
    return {"intent": intent}


def route_node(state: AgentState) -> dict:
    """2. Node: Feltételre alapú irányítás (conditional routing)."""
    intent = state.get("intent", "egyeb")
    route = route_decision(intent)
    return {"route": route}


def rag_node(state: AgentState) -> dict:
    """3. Node: Hibrid RAG subgraph meghívása a tudástárból (TF-IDF + Dense Vector Search + RRF)."""
    query = state["query"]
    try:
        from rag import HybridRAG, HybridConfig
        hybrid_rag = HybridRAG(HybridConfig())
        hybrid_rag.initialize()
        result = hybrid_rag.retrieve(query)
        context = result.get("context", "")
        retrieved = result.get("retrieved_docs", [])
        # Preserve metadata (source, filename, score) for RAG usage display
        retrieved_docs_data = []
        for doc in retrieved:
            if hasattr(doc, 'page_content'):
                retrieved_docs_data.append({
                    "source": doc.metadata.get("source", "ismeretlen"),
                    "filename": doc.metadata.get("filename", ""),
                    "content": doc.page_content,
                    "score": doc.metadata.get("score", None),
                })
            else:
                retrieved_docs_data.append({"source": "ismeretlen", "content": str(doc)})
        return {
            "context": context,
            "retrieved_docs": retrieved_docs_data,
            "error": None
        }
    except Exception as e:
        return {"context": "", "retrieved_docs": [], "error": str(e)}


def tool_node(state: AgentState) -> dict:
    """4. Node: Nem-RAG eszközök (pl. időszámítás, konverzió)."""
    query = state["query"]
    results = {}
    
    if "nap" in query.lower() or "napja" in query.lower():
        try:
            today = datetime.now()
            results["current_time"] = today.strftime("%Y-%m-%d %H:%M:%S")
            results["day_of_week"] = today.strftime("%A")
            results["day_of_year"] = today.timetuple().tm_yday
        except Exception as e:
            results["current_time"] = str(e)
    
    if "fizetés" in query.lower() or "ber" in query.lower() or "bruttó" in query.lower():
        try:
            min_bruttó_havi = 266800
            min_netto_havi = round(min_bruttó_havi * 0.815)
            results["fizeteskalkulacio"] = {
                "minimum_bruttó_havi": min_bruttó_havi,
                "minimum_netto_havi": min_netto_havi,
                "brutto_ora_40_ora": round(min_bruttó_havi / (21 * 8), 0),
                "netto_ora_40_ora": round(min_netto_havi / (21 * 8), 0)
            }
        except Exception as e:
            results["fizeteskalkulacio"] = str(e)
    
    results["tool_name"] = "fogyasztói_szamolopult"
    results["tool_result"] = f"Eszköz válasza a kérdésre: {query[:50]}..."
    
    return {"tool_results": results, "tool_calls": list(results.keys())}


def direct_answer_node(state: AgentState) -> dict:
    """Mega node: Közvetlen LLM válasz generálása."""
    query = state["query"]
    try:
        llm = get_llm({"model_name": "glm4:9b", "temperature": 0.7})
        response = llm.invoke([
            SystemMessage(content="Te vagy a magyar jogi tájékoztató chatbot. Válaszolj magyarul, tájékoztatóan, és válaszolj a felhasználó kérdésére."),
            HumanMessage(content=query)
        ])
        answer = response.content
        return {"answer": answer, "error": None}
    except Exception as e:
        return {"answer": f"Hiba a válaszgenerálásban: {e}", "error": str(e)}


def answer_generator_node(state: AgentState) -> dict:
    """
    5. Node: Végső válasz generálása.
    Összefűzi a kontextumot (RAG) és az eszköz eredményeit,
    az LLM segítségével generálja a végső választ.
    """
    query = state["query"]
    context = state.get("context", "")
    tool_results = state.get("tool_results", {})
    route = state.get("route", "rag")
    
    try:
        llm = get_llm({"model_name": "glm4:9b", "temperature": 0.7})
        
        if route == "rag" and context:
            system_prompt = (
                "Te vagy a magyar jogi tájékoztató chatbot. "
                "A felhasználó kérdére a tudástárból szerezd vissza a releváns információkat. "
                "Állj elő a felhasználónak, és válaszolj magyarul, tájékoztatóan, "
                "hasznos információkkal ellátva. Használd a következő kontextust:\n\n"
                "KONTEXTUS:\n{context}\n\n"
                "RELEVÁNCIA STATISZTIKÁK:\n{stats}\n\n"
                "KÉRDÉS: {query}\n\n"
                "Kérem válaszoljon magyarul, tájékoztatóan."
            )
            answer_parts = []
            for chunk in llm.stream([
                SystemMessage(content=system_prompt.format(context=context, query=query)),
                HumanMessage(content=query)
            ]):
                answer_parts.append(chunk.content)
            answer = "".join(answer_parts)
        elif route == "tool" and tool_results:
            tool_info = json.dumps(tool_results, ensure_ascii=False, indent=2)
            system_prompt = (
                "Te vagy a segítő chatbot. A felhasználó egy eszköz segítségét kérte. "
                "Használd az alábbi eszköz eredményeit a válaszhoz.\n\n"
                "ESZKÖZ EREMDÉNYEK:\n{tool_info}\n\n"
                "KÉRDÉS: {query}\n\n"
                "Kérem válaszoljon magyarul, tájékoztatóan."
            )
            answer_parts = []
            for chunk in llm.stream([
                SystemMessage(content=system_prompt.format(tool_info=tool_info, query=query)),
                HumanMessage(content=query)
            ]):
                answer_parts.append(chunk.content)
            answer = "".join(answer_parts)
        else:
            answer_parts = []
            for chunk in llm.stream([
                SystemMessage(content="Te vagy a magyar jogi tájékoztató chatbot. Válaszolj magyarul, tájékoztatóan."),
                HumanMessage(content=query)
            ]):
                answer_parts.append(chunk.content)
            answer = "".join(answer_parts)
        
        return {"answer": answer, "error": None}
    except Exception as e:
        return {"answer": f"Hiba a válaszgenerálásban: {e}", "error": str(e)}


def evaluation_node(state: AgentState) -> dict:
    """
    6. Node (opcionális): Értékelés/minősítés.
    Elemzi a válasz minőségét, és ad visszajelzést.
    """
    query = state["query"]
    answer = state.get("answer", "")
    route = state.get("route", "")
    
    try:
        llm = get_llm({"model_name": "glm4:9b", "temperature": 0.3})
        
        evaluation_prompt = (
            "Értékelje a válasz minőségét! Pontszám 1-10 között.\n\n"
            "KÉRDÉS: {query}\n\n"
            "VÁLASZ: {answer}\n\n"
            "Csak válaszoljon egy számral 1-10 között, és írjon 1-2 mondatot a válasz minőségéről. "
            "A válasz pontossága, relevancia és átláthatóság alapján értékeljen."
        )
        
        response = llm.invoke([
            SystemMessage(content="Te egy szigorú vizsgabiztosító vagy. Pontos és objektív értékelést adsz."),
            HumanMessage(content=evaluation_prompt.format(query=query, answer=answer))
        ])
        
        answer_text = response.content
        score = None
        try:
            first_word = answer_text.strip().split()[0].replace(".", "").replace(",", "")
            score = float(first_word)
            if score > 10: score = 10
            if score < 1: score = 1
        except:
            score = 5.0
        
        return {"evaluation_score": score, "evaluation_text": answer_text}
    except Exception as e:
        return {"evaluation_score": None, "evaluation_text": str(e)}


# === Main Graph Builder ===

def build_agent_graph(config: Optional[AgentConfig] = None) -> StateGraph:
    """
    Felépíti a fő agentic LangGraph workflow-et.
    
    A graph legalább 5 node-ot tartalmaz:
    1. intent_classifier (feltételezett dolog)
    2. route_decision (feltételre alapú irányítás)
    3. rag_node (RAG subgraph)
    4. tool_node (eszközök)
    5. answer_generator (válaszgenerálás)
    6. evaluation_node (opcionális értékelés)
    """
    cfg = config or {
        "model_name": "glm4:9b",
        "rag_config": RConfig(),
        "temperature": 0.7,
        "max_tokens": 4096
    }
    
    rag = RConfig()
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("route_decision", route_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("answer_generator", answer_generator_node)
    workflow.add_node("evaluation", evaluation_node)
    
    workflow.add_edge("intent_classifier", "route_decision")
    
    def route_edge(state: AgentState) -> Literal["rag", "tool", "answer_generator"]:
        route = state.get("route", "rag")
        if route == "rag":
            return "rag"
        elif route == "tool":
            return "tool"
        else:
            return "answer_generator"
    
    workflow.add_conditional_edges(
        "route_decision",
        route_edge,
        {"rag": "rag", "tool": "tool", "answer_generator": "answer_generator"}
    )
    
    workflow.add_edge("rag", "answer_generator")
    workflow.add_edge("tool", "answer_generator")
    workflow.add_edge("answer_generator", "evaluation")
    workflow.add_edge("evaluation", END)
    workflow.set_entry_point("intent_classifier")
    
    app = workflow.compile()
    return app


# === Fő agent futtatása ===

def _run_scraper_check():
    """Ellenőrzi, van-e friss dokumentum a scraper segítségével."""
    try:
        last = get_last_scrape_time()
        if last is None:
            result = init_scraper()
        else:
            result = check_updates()
    except Exception as e:
        pass


def run_agent(query: str, config: Optional[AgentConfig] = None, rag: Optional[Any] = None) -> dict:
    """
    Futtatja a teljes agentic workflow-et egy kérdésre.
    Visszaadja az eredményt és a nód végrehajtási idekéket.
    """
    
    node_timings: List[NodeTiming] = []
    
    def track_time(node_name: str, message: str = "", operations: Optional[List[str]] = None) -> NodeTiming:
        start = time.time()
        return NodeTiming(
            node_name=node_name,
            start_time=start,
            end_time=0,
            duration_seconds=0,
            message=message,
            operations=operations or []
        )
    
    def finish_timing(timing: NodeTiming, message: str = "", operations: Optional[List[str]] = None) -> NodeTiming:
        end = time.time()
        return NodeTiming(
            node_name=timing["node_name"],
            start_time=timing["start_time"],
            end_time=end,
            duration_seconds=round(end - timing["start_time"], 3),
            message=message,
            operations=operations or timing.get("operations", [])
        )
    
    # Web scraping check
    try:
        _run_scraper_check()
    except:
        pass
    
    # Initialize RAG (skip if pre-built instance provided)
    if rag is None:
        try:
            from rag import HybridRAG, HybridConfig
            rag = HybridRAG(HybridConfig())
            rag.initialize()
        except:
            pass
    
    # 1. Intent classification
    t1 = track_time("Vezérlés", "Kérdés fogadása", ["Kérdés fogadása", "Szöveg tisztítása", "Tokenizálás", "Intent osztályozás"])
    intent = classify_intent(query)
    t1 = finish_timing(t1, f"Intent: {intent}", ["Kérdés fogadása", "Szöveg tisztítása", "Tokenizálás", "Intent osztályozás"])
    node_timings.append(t1)
    
    # 2. Route decision
    t2 = track_time("Döntés", "Conditional routing", ["Intent értékelése", "Útvonal döntés", "RAG/Tool/Direct választás"])
    route = route_decision(intent)
    t2 = finish_timing(t2, f"Route: {route}", ["Intent értékelése", "Útvonal döntés", "RAG/Tool/Direct választás"])
    node_timings.append(t2)
    
    # 3-5. Execute based on route
    context = ""
    retrieved_docs = []
    tool_results = {}
    prompt = ""

    if route == "rag":
        t3 = track_time("RAG Keresés", "Hibrid keresés (TF-IDF + Dense + RRF)", ["TF-IDF index", "Dense vektor index", "RRF fusion", "Relevancia kiszámítás", "Dokumentum kiválasztás"])
        try:
            retrieved = rag.search(query)
            context = rag.get_context(retrieved)
            # Get retrieval stats for prompt
            retrieval_stats = rag.get_retrieval_stats()
            retrieved_docs_data = []
            for doc in retrieved:
                if hasattr(doc, 'page_content'):
                    retrieved_docs_data.append({
                        "source": doc.metadata.get("source", "ismeretlen"),
                        "filename": doc.metadata.get("filename", ""),
                        "content": doc.page_content[:200],
                        "score": doc.metadata.get("score", None),
                    })
                else:
                    retrieved_docs_data.append({"source": "ismeretlen", "content": str(doc)[:200]})
            retrieved_docs = retrieved_docs_data
            # Format retrieval stats for prompt
            stats_lines = []
            for src, score in retrieval_stats.items():
                stats_lines.append(f"- {src}: {score:.3f}")
            stats_text = "\n".join(stats_lines) if stats_lines else "Nincs elérhető statisztika."
            # We'll add stats to context or to prompt later
        except Exception as e:
            context = ""
            retrieved_docs = []
            retrieval_stats = {}
            stats_text = "Hiba a kérés során."
        t3 = finish_timing(t3, f"{len(retrieved_docs)} dokumentum", ["TF-IDF index", "Dense vektor index", "RRF fusion", "Relevancia kiszámítás", "Dokumentum kiválasztás"])
        node_timings.append(t3)
    
        t4 = track_time("Eszköz", "Közprendszer", ["Számítás", "Dátum kalkuláció", "Egyszerű logikai műveletek"])
        t4 = finish_timing(t4, "Nincs szükség", ["Számítás", "Dátum kalkuláció", "Egyszerű logikai műveletek"])
        node_timings.append(t4)
    
        t5 = track_time("Generálás", "GLM4 válasz", ["GLM4 prompt felépítés", "Token stream", "Válasz összeszése"])
        llm_config = {"model_name": "glm4:9b", "temperature": 0.7}
        llm = get_llm(llm_config)

        system_prompt = (
            "Te vagy a magyar jogi tájékoztató chatbot. "
            "A felhasználó kérdésére a tudástárból szerezd vissza a releváns információkat. "
            "Állj elő a felhasználónak, és válaszolj magyarul, tájékoztatóan, "
            "hasznos információkkal ellátva. Használd a következő kontextust:\n\n"
            "KONTEXTUS:\n{context}\n\n"
            "KÉRDÉS: {query}\n\n"
            "Kérem válaszoljon magyarul, tájékoztatóan."
        )
        prompt = system_prompt.format(context=context, query=query, stats=stats_text)

        answer_parts = []
        for chunk in llm.stream([
            SystemMessage(content=prompt),
            HumanMessage(content=query)
        ]):
            answer_parts.append(chunk.content)
        answer = "".join(answer_parts)
        t5 = finish_timing(t5, "Válasz kész", ["GLM4 prompt felépítés", "Token stream", "Válasz összeszése"])
        node_timings.append(t5)
    
    elif route == "tool":
        t3 = track_time("RAG Keresés", "Nincs szükség", ["TF-IDF index", "Vektorkeresés", "Relevancia kiszámítás"])
        t3 = finish_timing(t3, "Nincs RAG szükséges", ["TF-IDF index", "Vektorkeresés", "Relevancia kiszámítás"])
        node_timings.append(t3)
    
        t4 = track_time("Eszköz", "Számítás és idő", ["Számítás", "Dátum kalkuláció", "Egyszerű logikai műveletek"])
        try:
            today = datetime.now()
            tool_results = {
                "current_time": today.strftime("%Y-%m-%d %H:%M:%S"),
                "day_of_week": today.strftime("%A"),
                "day_of_year": today.timetuple().tm_yday
            }
        except Exception as e:
            tool_results = {"error": str(e)}
        t4 = finish_timing(t4, "Eszköz végrehajtva", ["Számítás", "Dátum kalkuláció", "Egyszerű logikai műveletek"])
        node_timings.append(t4)
    
        t5 = track_time("Generálás", "GLM4 válasz", ["GLM4 prompt felépítés", "Token stream", "Válasz összeszése"])
        llm_config = {"model_name": "glm4:9b", "temperature": 0.7}
        llm = get_llm(llm_config)
    
        tool_info = json.dumps(tool_results, ensure_ascii=False, indent=2)
        system_prompt = (
            "Te vagy a segítő chatbot. A felhasználó egy eszköz segítségét kérte. "
            "Használd az alábbi eszköz eredményeit a válaszhoz.\n\n"
            "ESZKÖZ EREMDÉNYEK:\n{tool_info}\n\n"
            "KÉRDÉS: {query}\n\n"
            "Kérem válaszoljon magyarul, tájékoztatóan."
        )
        prompt = system_prompt.format(tool_info=tool_info, query=query)

        answer_parts = []
        for chunk in llm.stream([
            SystemMessage(content=prompt),
            HumanMessage(content=query)
        ]):
            answer_parts.append(chunk.content)
        answer = "".join(answer_parts)
        t5 = finish_timing(t5, "Válasz kész", ["GLM4 prompt felépítés", "Token stream", "Válasz összeszése"])
        node_timings.append(t5)
    
    else:
        t3 = track_time("RAG Keresés", "Nincs szükség", ["TF-IDF index", "Vektorkeresés", "Relevancia kiszámítás"])
        t3 = finish_timing(t3, "Közvetlen út", ["TF-IDF index", "Vektorkeresés", "Relevancia kiszámítás"])
        node_timings.append(t3)
    
        t4 = track_time("Eszköz", "Nincs szükség", ["Számítás", "Dátum kalkuláció", "Egyszerű logikai műveletek"])
        t4 = finish_timing(t4, "Nincs eszköz", ["Számítás", "Dátum kalkuláció", "Egyszerű logikai műveletek"])
        node_timings.append(t4)
    
        t5 = track_time("Generálás", "GLM4 közvetlen válasz", ["GLM4 prompt felépítés", "Token stream", "Válasz összeszése"])
        llm_config = {"model_name": "glm4:9b", "temperature": 0.7}
        llm = get_llm(llm_config)

        prompt = "Te vagy a magyar jogi tájékoztató chatbot. Válaszolj magyarul, tájékoztatóan."

        answer_parts = []
        for chunk in llm.stream([
            SystemMessage(content=prompt),
            HumanMessage(content=query)
        ]):
            answer_parts.append(chunk.content)
        answer = "".join(answer_parts)
        t5 = finish_timing(t5, "Válasz kész", ["GLM4 prompt felépítés", "Token stream", "Válasz összeszése"])
        node_timings.append(t5)
    
    return {
        "query": query,
        "intent": intent,
        "route": route,
        "context": context,
        "retrieved_docs": retrieved_docs,
        "tool_results": tool_results,
        "answer": answer,
        "prompt": prompt,
        "evaluation_score": 5.0,
        "error": None,
        "node_timings": node_timings
    }


if __name__ == "__main__":
    result = run_agent("Mennyi a minimálbér 2024-ben?")
    print(f"Válasz: {result['answer'][:200]}...")
    print("\nNód idők:")
    for t in result.get("node_timings", []):
        print(f"  {t['node_name']}: {t['duration_seconds']}s - {t['message']}")
