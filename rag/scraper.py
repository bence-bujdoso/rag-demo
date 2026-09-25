"""Helyi dokumentum scraping modul.

A dokumentumok helyben vannak storage-ban (data/web_results.json).
Az első indulásnál feltölti a seed adatokat, majd csak ellenőrzi a frissítést.

Használat:
    from rag.scraper import scrape_documents, check_updates, init_scraper, get_web_results
    
    # Első indításnál
    init_scraper()  # feltölti a web_results.json-t ha nem létezik
    
    # Utánnal csak frissítés
    check_updates()  # 1 óránként ellenőrzi
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict

# Konfiguráció
DATA_DIR = "/home/columbo/ExtData/AIprojects/RagDemo/data"
WEB_RESULTS_FILE = os.path.join(DATA_DIR, "web_results.json")
LAST_CHECK_FILE = os.path.join(DATA_DIR, ".last_check")
CHECK_INTERVAL_HOURS = 1  # Maximum 1 óra közötti frissítés

# Keresési lekérdezések a magyar jogi dokumentumokhoz
SEARCH_QUERIES = [
    "magyar munkaügyi törvény 2026",
    "magyar adószabályok 2026",
    "magyar lakhatági jogok 2026",
    "magyar fogyasztóvédelmi törvény 2026",
    "magyar munkaerőpiac 2026",
    "magyar állampolgársági jogok 2026",
]


def _get_signature(content: str) -> str:
    """Számít hash-t a tartalom azonosításához frissítés ellenőrzéshez."""
    return hashlib.md5(content.encode()).hexdigest()


def _seed_web_results() -> int:
    """Ha nincs web_results.json, feltöltjük alapértelmezett adatokkal.
    
    Az első indításnál fut, amikor nincs még adat.
    A felhasználó később frissítheti a web_results.json-t manuálisan.
    """
    if os.path.exists(WEB_RESULTS_FILE):
        return len(get_web_results())
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    seed_data = [
        {
            "title": "Munkaszerződési Törvény (1992. évi I. törvény) - Magyar Közlöny",
            "link": "https://www.magyarjogtar.haszoninfo.hu/tortenet/tort001.htm",
            "snippet": "A munkaszerződési jog alapelvei, a próbaidő, a fizetés és a munkafeltételek szabályai.",
            "displayLink": "magyarjogtar.haszoninfo.hu",
            "formattedUrl": "https://www.magyarjogtar.haszoninfo.hu",
            "query": "magyar munkaügyi törvény 2026",
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Szja-törvény (1996. évi XCVI. törvény) - Adózási szabályok",
            "link": "https://www.akh.hu/sajtoszoba/sajto-tudastar/sajto-tudastar/szja-tortenet.html",
            "snippet": "A személyi jövedelemadó szabályai, szabadalmi díjak, nyugdíjpénztárak.",
            "displayLink": "akh.hu",
            "formattedUrl": "https://www.akh.hu",
            "query": "magyar adószabályok 2026",
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Lakásügyi törvény (1993. évi LXV. törvény) - Lakhatási jogok",
            "link": "https://www.kormany.hu/lakhatas-jogok",
            "snippet": "A lakbér, a bérleti szerződés és a lakhatási jogok szabályai.",
            "displayLink": "kormany.hu",
            "formattedUrl": "https://www.kormany.hu",
            "query": "magyar lakhatági jogok 2026",
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Fogyasztóvédelmi törvény (1997. évi LXXXIV. törvény)",
            "link": "https://www.nmvh.hu/fogyasztoverdelem",
            "snippet": "A fogyasztóvédelmi jogok, garancia, távolságos vásárlás és visszafizetés.",
            "displayLink": "nmvh.hu",
            "formattedUrl": "https://www.nmvh.hu",
            "query": "magyar fogyasztóvédelmi törvény 2026",
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Munkaerőpiac és foglalkoztatás 2026 - KSH Adatok",
            "link": "https://www.ksh.hu/munkaeropiac",
            "snippet": "A magyar munkaerőpiac statisztikái, állások, munkanélküliségi ráta.",
            "displayLink": "ksh.hu",
            "formattedUrl": "https://www.ksh.hu",
            "query": "magyar munkaerőpiac 2026",
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Tájékoztató a munkaügyi bírósági eljárásról",
            "link": "https://www.bk.gov.hu/munkaügyi-biral",
            "snippet": "Munkaügyi bírósági határidők, per indítása, döntések érvényesítése.",
            "displayLink": "bk.gov.hu",
            "formattedUrl": "https://www.bk.gov.hu",
            "query": "munkaügyi bírósági határidő 2026",
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Nettó és bruttó fizetés kalkulátor 2026",
            "link": "https://www.netto-brutto.hu/kalkulalo",
            "snippet": "Bruttó nettó átváltás, munkáltatói járulékok, szociális biztosítás.",
            "displayLink": "netto-brutto.hu",
            "formattedUrl": "https://www.netto-brutto.hu",
            "query": "nettó bruttó fizetés kalkuláció 2026",
            "timestamp": datetime.now().isoformat()
        },
        {
            "title": "Általános adóvedelem - GDPR és személyes adatok védelme",
            "link": "https://www.nikbps.hu/adatvedelem",
            "snippet": "Az adatvédelmi szabályok, GDPR betartása, személyes adatok kezelése.",
            "displayLink": "nikbps.hu",
            "formattedUrl": "https://www.nikbps.hu",
            "query": "magyar adóvedelem 2026",
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    with open(WEB_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(seed_data, f, indent=2, ensure_ascii=False)
    
    with open(LAST_CHECK_FILE, "w") as f:
        f.write(datetime.now().isoformat())
    
    pass  # print(f"[SCRAPER] ⚠️ Nincs web_results.json - seed adatok betöltve ({len(seed_data)} dokumentum)")
    pass  # print(f"[SCRAPER]   {WEB_RESULTS_FILE}-ba mentve")
    
    return len(seed_data)


def scrape_documents(query: Optional[str] = None, max_results: int = 10) -> Dict:
    """Helyi dokumentumokat keres és ad vissza.
    
    Nem futtat külső letöltést - a data/web_results.json fájlt használja.
    Ha a fájl nem létezik, seed adatokkal tölti fel.
    
    Args:
        query: Keresési kifejezés (nem használják, csak interfész kompatibilitás).
        max_results: Maximális eredmények száma (nem használják).
    
    Returns:
        Dictionary a keresési eredményeivel.
    """
    if not os.path.exists(WEB_RESULTS_FILE):
        count = _seed_web_results()
        return {
            "success": True,
            "count": count,
            "seed": True,
            "message": "Helyi seed adatok letöltve. A web_results.json tartalmazza a dokumentumokat.",
            "results": get_web_results(),
            "timestamp": datetime.now().isoformat()
        }
    
    results = get_web_results()
    
    # Ha query van, szűrjük rá
    if query:
        filtered = [r for r in results if query.lower() in r.get("title", "").lower() or query.lower() in r.get("snippet", "").lower()]
        return {
            "success": True,
            "count": len(filtered),
            "results": filtered,
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "success": True,
        "count": len(results),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


def check_updates(query: Optional[str] = None, max_results: int = 10) -> Dict:
    """Ellenőrzi, vannak-e frissebb dokumentumok az utolsó ellenőrzés óta.
    
    Csak akkor futtat scraper-t, ha 1 óránál tovább van az utolsó ellenőrzés.
    
    Returns:
        Dictionary a frissítés állapotáról.
    """
    if os.path.exists(LAST_CHECK_FILE):
        try:
            with open(LAST_CHECK_FILE, "r") as f:
                last_check_str = f.read().strip()
            last_check = datetime.fromisoformat(last_check_str)
            time_since = datetime.now() - last_check
            
            if time_since < timedelta(hours=CHECK_INTERVAL_HOURS):
                return {
                    "status": "cached",
                    "message": f"Az utolsó ellenőrzés {time_since.seconds // 60} perce volt. Újraellenőrzés: {CHECK_INTERVAL_HOURS}óránként.",
                    "last_check": last_check_str,
                    "next_check": (last_check + timedelta(hours=CHECK_INTERVAL_HOURS)).isoformat()
                }
        except (ValueError, IOError):
            pass
    
    # Nincs megadva az utolsó ellenőrzés, vagy eltelt 1 óra - futtatjuk a scraper-t
    return scrape_documents(query=query, max_results=max_results)


def init_scraper(query: Optional[str] = None, max_results: int = 10) -> Dict:
    """Az első futásnál letölti a dokumentumokat.
    
    Ha nincs web_results.json, seed adatokkal tölti ki.
    Ha már van, visszaadja az aktuális állapotot.
    
    Használat:
        from rag.scraper import init_scraper
        result = init_scraper()  # Első indításnál
    
    Returns:
        Dictionary a letöltési eredményeivel.
    """
    if not os.path.exists(WEB_RESULTS_FILE):
        count = _seed_web_results()
        return {
            "success": True,
            "count": count,
            "seed": True,
            "message": "Seed adatok letöltve.",
            "timestamp": datetime.now().isoformat()
        }
    
    result = scrape_documents(query=query, max_results=max_results)
    
    # Frissítés idejének mentése
    if result.get("success"):
        with open(LAST_CHECK_FILE, "w") as f:
            f.write(datetime.now().isoformat())
        with open(os.path.join(DATA_DIR, ".last_scrape"), "w") as f:
            f.write(datetime.now().isoformat())
    
    return result


def get_last_scrape_time() -> Optional[datetime]:
    """Visszaadja az utolsó scraping időpontját."""
    scrape_file = os.path.join(DATA_DIR, ".last_scrape")
    if os.path.exists(scrape_file):
        try:
            with open(scrape_file, "r") as f:
                return datetime.fromisoformat(f.read().strip())
        except (ValueError, IOError):
            return None
    return None


def get_web_results() -> List[Dict]:
    """Betölti a web_results.json fájlt."""
    if os.path.exists(WEB_RESULTS_FILE):
        try:
            with open(WEB_RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (ValueError, IOError):
            return []
    return []


if __name__ == "__main__":
    pass  # print("🔍 RagDemo Helyi Dokumentum Scraper")
    pass  # print(f"   Web results: {'✓ Van' if os.path.exists(WEB_RESULTS_FILE) else '✗ Nincs'}")
    pass  # print(f"   Dokumentumok: {len(get_web_results())}")
    pass  # print()
    pass  # print("Használat:")
    pass  # print("  from rag.scraper import init_scraper, check_updates")
    pass  # print("  init_scraper()   # Első indításnál")
    pass  # print("  check_updates()  # Utánnal (1 óránként)")
