"""Egyszerűsített terhelési teszt (Load Test) az Agentic RAG chatbot-ra.

A teszt 50-200 lekérdezést küld a rendszernek, és összegzi:
- Alapvető latency metrikák (átlagos, min, max, percentile)
- A rendszer fő szűk keresztmetszetének azonosítása (bottleneck)
- 1-2 konkrét optimalizálási javaslat

Használat:
    python tests/load_test.py --queries 50
    python tests/load_test.py --queries 200
    python tests/load_test.py --queries 100 --duration 300
"""

import time
import statistics
import argparse
from datetime import datetime
from pathlib import Path
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import run_agent


def generate_test_queries(count: int = 50) -> list:
    """Generál teszt kérdéseket a különböző kategóriákból."""
    queries = [
        "Mennyi a munkaszerződés próbaideje?",
        "Mi a minimálbér 2024-ben?",
        "Hány napos szabadság jogos a munkavállalónak?",
        "Mennyi az ÁFA Magyarországon?",
        "Mi a személyi jövedelemadó mértéke?",
        "Hány napig lehet online vásárlásból visszavenni?",
        "Mi az adatvédelmi rendelet (GDPR) célja?",
        "Mennyi a nyugdíjbaj járulék?",
        "Hány napos elbocsátási jog van?",
        "Mikor kell adót fizetni év elején?",
        "Milyen támogatást kap az új munkahely?",
        "Mi az Otthonkereső Program?",
        "Mennyi a munkaerőpiaci járulékok összeg?",
        "Hány éves korig jár nyugdíjba?",
        "Milyen jogok vannak a terhes munkavállalónak?",
        "Mennyit fizet a munkáltató a nyugdíjbaj járulékban?",
        "Hány napos próbaidő van a szerződésben?",
        "Mi az adatvédelmi irányelv (DPO)?",
        "Hány éves korig lehet szociális lakat kérni?",
        "Mennyi a fogyasztói elbocsátási jog ideje?",
    ] * (count // 20 + 1)
    
    return queries[:count]


def run_load_test(num_queries: int = 50, output_file: str = "load_test_results.json") -> dict:
    """
    Futtatja a terhelési tesztet.
    
    Args:
        num_queries: A lekérdezések száma (50-200)
        output_file: Az eredmények mentési fájlja
    
    Returns:
        Az összesített eredmények dictionary-ként
    """
    print(f"=" * 60)
    print(f"TERHELÉSI TESZT - {num_queries} lekérdezés")
    print(f"Kezdési idő: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 60)
    
    queries = generate_test_queries(num_queries)
    results = {
        "metadata": {
            "num_queries": num_queries,
            "start_time": datetime.now().isoformat(),
            "queries": queries
        },
        "latencies": [],
        "successes": 0,
        "errors": 0,
        "error_details": [],
        "route_distribution": {"rag": 0, "tool": 0, "direct": 0, "error": 0}
    }
    
    for i, query in enumerate(queries, 1):
        try:
            start = time.perf_counter()
            
            result = run_agent(query)
            
            end = time.perf_counter()
            latency = end - start
            
            results["latencies"].append(latency)
            results["successes"] += 1
            
            route = result.get("route", "error")
            results["route_distribution"][route] = results["route_distribution"].get(route, 0) + 1
            
            # Nyomtatás minden 10. kérdésnél
            if i % 10 == 0:
                print(f"  [{i}/{num_queries}] {latency:.2f}s | Intent: {result.get('intent')} | Route: {route}")
                
        except Exception as e:
            results["errors"] += 1
            results["error_details"].append({"query": query, "error": str(e)})
            results["route_distribution"]["error"] += 1
            print(f"  [{i}/{num_queries}] HIBA: {str(e)[:50]}...")
    
    # Statisztikák kiszámítása
    latencies = results["latencies"]
    if latencies:
        results["stats"] = {
            "count": len(latencies),
            "avg_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
            "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0],
        }
    else:
        results["stats"] = {"count": 0}
    
    # Bemeneti adatok
    results["end_time"] = datetime.now().isoformat()
    results["duration_seconds"] = (datetime.now() - datetime.fromisoformat(results["metadata"]["start_time"])).total_seconds()
    
    # Teszt eredmények
    results["bottleneck_analysis"] = analyze_bottleneck(results)
    results["optimization_suggestions"] = generate_optimizations(results)
    
    # Mentés fájlba
    output_path = Path("tests") / output_file
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"TERHELÉSI TESZT LEZÁRULVA")
    print(f"{'=' * 60}")
    print_summary(results)
    
    return results


def analyze_bottleneck(results: dict) -> dict:
    """Azonosítja a rendszer fő szűk keresztmetszetét."""
    stats = results.get("stats", {})
    latencies = results.get("latencies", [])
    route_dist = results.get("route_distribution", {})
    
    bottleneck = {
        "primary": "N/A",
        "severity": "low",
        "details": []
    }
    
    # 1. LLM hívás latency
    avg_latency = stats.get("avg_latency", 0)
    if avg_latency > 30:
        bottleneck["primary"] = "LLM (Ollama) válaszidő"
        bottleneck["severity"] = "high"
        bottleneck["details"].append(
            f"Az átlagos válaszidő ({avg_latency:.1f}s) magas. "
            f"A glm4:9b modell nagyobb a CPU-n."
        )
    
    # 2. RAG indexelés
    if latencies and max(latencies) > 120:
        bottleneck["primary"] = "RAG indexelés (első kérés)"
        bottleneck["severity"] = "medium"
        bottleneck["details"].append(
            "Az első lekérdezésnél a FAISS indexelés magas latenciát OKOz. "
            "Ennél a hálóban nem futhat újra."
        )
    
    # 3. Route distribution
    rag_ratio = route_dist.get("rag", 0) / max(results.get("successes", 1), 1)
    if rag_ratio > 0.8:
        bottleneck["primary"] = "RAG túlzott használata"
        bottleneck["severity"] = "medium"
        bottleneck["details"].append(
            f"A {rag_ratio*100:.0f}% kérés RAG-t használ. "
            f"Ez azt jelzi, hogy az agent túl gyakran hívja a tudástárt. "
            f"Optimalizálható az irányítási logikával."
        )
    
    # 4. Memory
    bottleneck["details"].append(
        "A rendszer memory-ot használt a FAISS indexeléséhez. "
        "A 6 dokumentum kiskömben van, de több esetén bővíteni kell."
    )
    
    return bottleneck


def generate_optimizations(results: dict) -> list:
    """1-2 konkrét optimalizálási javaslat."""
    stats = results.get("stats", {})
    avg_latency = stats.get("avg_latency", 0)
    
    suggestions = []
    
    # Javaslat 1: Ollama model optimalizálás
    if avg_latency > 20:
        suggestions.append({
            "priority": "high",
            "category": "LLM Performance",
            "suggestion": "Használjunk egy kisebb, gyorsabb modellt az Ollama-ban",
            "details": (
                "A glm4:9b (9.4B parameter) lassú CPU-n. "
                "Alternatíva: qwen3vl-8b vagy még kisebb, 3-4B parameterű modell "
                "(pl. 'tinyllama', 'phi-2'). "
                "A csere jelentős latency-csökkentést eredményezhet (30s -> 8-12s)."
            ),
            "expected_improvement": "50-70% latency reduction"
        })
    
    # Javaslat 2: FAISS index caching
    suggestions.append({
        "priority": "medium",
        "category": "RAG Optimization",
        "suggestion": "FAISS indexelés caching és előtelepítés",
        "details": (
            "Az első kérésnél a FAISS index felépítése hosszú (10-30s). "
            "Optimalizáció: index előtelepítése Docker konténer indításakor, "
            "és memory-ben tartása. Ezenkívül a index mentése és újrahasználata "
            "docker-compose volume-vel."
        ),
        "expected_improvement": "Eliminates first-request latency"
    })
    
    # Javaslat 3: Batch processing
    suggestions.append({
        "priority": "medium",
        "category": "Architecture",
        "suggestion": "Batch query processing és aszinkron LLM hívások",
        "details": (
            "Jelenleg minden kérés egy külön LLM hívást eredményez. "
            "Optimalizáció: batch processing, ahol több kérést egyszerre "
            "feldolgoz az LLM. Ez különösen hasznos nagyobb terhelések esetén."
        ),
        "expected_improvement": "Throughput increase: 2-3x"
    })
    
    return suggestions


def print_summary(results: dict) -> None:
    """Nyomtatja a teszt eredményeinek összefoglalóját."""
    stats = results.get("stats", {})
    
    print(f"\n📊 ALAPVETŐ STATISZTIKÁK:")
    print(f"  Lekérdezések száma: {stats.get('count', 0)}/{results['metadata']['num_queries']}")
    print(f"  Sikeres: {results['successes']}, Hibás: {results['errors']}")
    print(f"  Időfutam: {results.get('duration_seconds', 0):.1f}s")
    
    if stats:
        print(f"\n⏱️ LATENCY METRIKÁK:")
        print(f"  Átlag: {stats.get('avg_latency', 0):.2f}s")
        print(f"  Median: {stats.get('median_latency', 0):.2f}s")
        print(f"  Min: {stats.get('min_latency', 0):.2f}s")
        print(f"  Max: {stats.get('max_latency', 0):.2f}s")
        print(f"  Std: {stats.get('std_latency', 0):.2f}s")
        print(f"  P95: {stats.get('p95_latency', 0):.2f}s")
        print(f"  P99: {stats.get('p99_latency', 0):.2f}s")
    
    print(f"\n🔀 ROUTE ELOSZLÁS:")
    for route, count in results.get("route_distribution", {}).items():
        print(f"  {route}: {count}")
    
    print(f"\n🔍 BOTTLENECK ANALÍZIS:")
    bottleneck = results.get("bottleneck_analysis", {})
    print(f"  Fő szűk keresztmetszet: {bottleneck.get('primary', 'N/A')}")
    print(f"  Szélsőségszint: {bottleneck.get('severity', 'N/A')}")
    print(f"  Részletek:")
    for detail in bottleneck.get("details", []):
        print(f"    - {detail}")
    
    print(f"\n⚡ OPTIMALIZÁLÁSI JAVASLATOK:")
    for i, sug in enumerate(results.get("optimization_suggestions", []), 1):
        print(f"  {i}. [{sug['priority'].upper()}] {sug['category']}: {sug['suggestion']}")
        print(f"     Várt javulás: {sug.get('expected_improvement', 'N/A')}")
    
    print(f"\n📄 Eredmények mentése: tests/load_test_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RAG Load Test")
    parser.add_argument("--queries", type=int, default=50, help="Lekérdezések száma (50-200)")
    parser.add_argument("--output", type=str, default="load_test_results.json", help="Eredmény fájl neve")
    parser.add_argument("--dry-run", action="store_true", help="Csak a kérdések generálása, futtatás nélkül")
    
    args = parser.parse_args()
    
    if args.dry_run:
        queries = generate_test_queries(args.queries)
        print(f"Generált {len(queries)} kérdés:")
        for i, q in enumerate(queries[:5], 1):
            print(f"  {i}. {q}")
        print(f"  ...")
    else:
        run_load_test(args.queries, args.output)
