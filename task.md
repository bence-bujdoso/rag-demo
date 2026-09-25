Migrációs Terv: TF-IDF Rendszer Átalakítása Hibrid RAG Architektúrává

Ez a dokumentum összefoglalja azokat a fejlesztési lépéseket, amelyek szükségesek egy meglévő, Python alapú TF-IDF (szórvány/sparse) keresőrendszer kibővítéséhez vektoros (sűrű/dense) kereséssel, ezáltal egy Hibrid RAG rendszert hozva létre.

1. Előkészületek és Technológia Választás
Mivel a TF-IDF már adott, a következő eszközöket kell kiválasztani a vektoros ághoz:
[ ] Embedding modell kiválasztása: Olyan modellre lesz szükség, ami a szövegeket számszerű vektorokká alakítja.
Nyílt forráskódú (lokális): sentence-transformers (pl. all-MiniLM-L6-v2 vagy többnyelvű modellek magyar nyelvhez, mint a paraphrase-multilingual-MiniLM-L12-v2).
[ ] Vektoradatbázis kiválasztása: A vektorok tárolására és a gyors hasonlóság-keresésre.
Kisebb projektekhez / memóriában futó: FAISS, ChromaDB.
Nagyobb, skálázható projektekhez: Qdrant, Pinecone, Weaviate, Milvus.

2. Az Adatfeldolgozási Csővezeték (Pipeline) Kibővítése
A jelenlegi rendszer valószínűleg tokenizálja a szöveget és TF-IDF mátrixot épít belőle. Ezt ki kell egészíteni a vektoros ággal.
[ ] Chunking (Szövegdarabolás) felülvizsgálata: A vektoros modellek bemeneti mérete korlátozott (pl. 512 token). Ellenőrizni kell, hogy az aktuális darabolási stratégia (chunking) megfelel-e a kiválasztott embedding modellnek.
[ ] Embeddingek generálása: Írni egy scriptet, amely a meglévő dokumentumokon végigfut, és a kiválasztott modellel legenerálja a vektorokat (dense vectors).
[ ] Indexelés: A vektorok és a hozzájuk tartozó metaadatok (vagy a dokumentum azonosítók) betöltése a kiválasztott vektoradatbázisba.

3. A Keresési (Retrieval) Logika Átalakítása
A meglévő keresőfüggvényt ketté kell bontani, majd párhuzamosítani.
[ ] TF-IDF Keresés: A meglévő függvény meghívása a felhasználói kérdésre (visszatér a top $K$ dokumentum azonosítójával és pontszámával).
[ ] Vektoros Keresés: A felhasználói kérdés vektorizálása (embedding), majd a vektoradatbázis lekérdezése (szintén visszatér a top $K$ dokumentummal és a koszinusz hasonlósági / távolsági pontszámmal).
[ ] Opcionális (Teljesítmény): A két lekérdezés aszinkron (párhuzamos) futtatása a válaszidő csökkentése érdekében (pl. asyncio használatával).

4. RRF (Reciprocal Rank Fusion) Implementálása
Mivel a TF-IDF és a vektoros keresés teljesen más skálán pontoz, a pontszámokat nem lehet egyszerűen összeadni. Helyette a helyezések (rank) alapján kell kombinálni őket.
[ ] RRF Algoritmus megírása Pythonban:
Bemenet: A két (TF-IDF és Vektoros) találati lista.
Képlet: $RRF(d) = \sum \frac{1}{k + rank(d)}$, ahol $d$ a dokumentum, $rank(d)$ a helyezése a listában, a $k$ pedig egy konstans (általában 60).
[ ] Dokumentumok Újrarangsorolása (Re-ranking): Az RRF pontszámok alapján a dokumentumok sorba rendezése, és a végső top $N$ dokumentum kiválasztása.

5. Integráció az LLM-mel (Generációs fázis)
Ha az RRF kiválasztotta a legjobb dokumentumokat, azokat továbbítani kell az LLM-nek.
[ ] Kontextus összeállítása: A nyertes dokumentumok szövegének összefűzése a promptba. (Ez a lépés valószínűleg már megvan a jelenlegi rendszerben, csak a bemeneti listát kell cserélni az RRF eredményére).
[ ] Prompt tesztelése: Ellenőrizni, hogy az LLM jól kezeli-e az új, valószínűleg sokszínűbb kontextust.

6. Finomhangolás és Értékelés
[ ] Alpha paraméter (Súlyozás) bevezetése: Lehetőséget biztosítani arra, hogy eltoljuk a hangsúlyt. Pl. alpha * vektor_pont + (1-alpha) * kulcsszó_pont (ha nem az RRF-et használjuk, hanem normalizált pontszámokat).
[ ] A/B Tesztelés: Néhány tesztkérdés lefuttatása csak a TF-IDF-fel, és a Hibrid megoldással. Az LLM válaszainak minőségének összehasonlítása.