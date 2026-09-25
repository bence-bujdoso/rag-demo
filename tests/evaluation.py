"""Értékelési készlet - 20 kérdés a magyar jogi tájékoztatóhoz.

A kérdések a következő témákra összpontosítanak:
1. Munkaszerződés és munkajogi kérdések
2. Adózás és szociális járulékok
3. Lakhatási jogok
4. Adatvédelmi jogok (GDPR)
5. Fogyasztóvédelmi jogok
6. Munkaerő-piaci kérdések
7. Általános tudásbázis kérdések
"""

EVALUATION_QUESTIONS = [
    # === Munkaszerződés (4 kérdés) ===
    {
        "id": 1,
        "question": "Mennyi a munkaszerződés próbaideje maximálisan?",
        "expected_keywords": ["próbaidő", "3 hónap", "három hónap", "3 hónapos"],
        "category": "munkajog"
    },
    {
        "id": 2,
        "question": "Mi a minimálbér 2026-ben bruttó?",
        "expected_keywords": ["266", "266800", "266.800", "minimálbér", "bruttó"],
        "category": "munkajog"
    },
    {
        "id": 3,
        "question": "Mennyit kap egy munkavállaló éves szabadságidejét?",
        "expected_keywords": ["20 nap", "20 napos", "szabadság", "nyaralás"],
        "category": "munkajog"
    },
    {
        "id": 4,
        "question": "Mikor szabad felmondani a munkaszerződést próbaidőben?",
        "expected_keywords": ["7 nap", "heti", "felmondás", "próbaidő"],
        "category": "munkajog"
    },
    
    # === Adózás (3 kérdés) ===
    {
        "id": 5,
        "question": "Mi az ÁFA (Általános Forgalmi Adó) mértéke Magyarországon?",
        "expected_keywords": ["27%", "27 százalék", "afa", "áfa mértéke", "27"],
        "category": "adozás"
    },
    {
        "id": 6,
        "question": "Mennyi a személyi jövedelemadó (SZJA) mértéke?",
        "expected_keywords": ["15%", "15 százalék", "szja", "személyi jövedelemadó", "15"],
        "category": "adozás"
    },
    {
        "id": 7,
        "question": "Mikor kell az adórendészeti nyilatkozatot benyújtani?",
        "expected_keywords": ["március", "20", "év eleji", "adórendészeti", "előleg"],
        "category": "adozás"
    },
    
    # === Lakhatás (3 kérdés) ===
    {
        "id": 8,
        "question": "Mi az Otthonkereső Program célja?",
        "expected_keywords": ["otthonkereső", "lakás beszerzés", "család", "támogatás"],
        "category": "lakhatasag"
    },
    {
        "id": 9,
        "question": "Milyen díjak tartoznak egy lakbérleti szerződésben?",
        "expected_keywords": ["lakbér", "fűtés", "víz", "csatorna", "díjak"],
        "category": "lakhatasag"
    },
    {
        "id": 10,
        "question": "Mennyi a lakbérlő éventi szabadsága?",
        "expected_keywords": ["20 nap", "szabadság", "lakbérlő", "évi"],
        "category": "lakhatasag"
    },
    
    # === Adatvédelem (3 kérdés) ===
    {
        "id": 11,
        "question": "Mi az általános adatelkülönítési rendelet (GDPR)?",
        "expected_keywords": ["GDPR", "2016/679", "adatelkülönítés", "2018", "megálma"],
        "category": "adoavedelem"
    },
    {
        "id": 12,
        "question": "Mennyi az adatvédelmi hatósági bírság a GDPR megsértéséért?",
        "expected_keywords": ["10 millió", "20 millió", "euró", "büntetés", "NMHH"],
        "category": "adoavedelem"
    },
    {
        "id": 13,
        "question": "Mennyi időben kell a adatvédelmi hatóságot értesíteni egy adatvédelmi beavatkozásról?",
        "expected_keywords": ["72 óra", "72", "hatóság", "beavatkozás", "adatvédelmi"],
        "category": "adoavedelem"
    },
    
    # === Fogyasztóvédelem (3 kérdés) ===
    {
        "id": 14,
        "question": "Hány napos elbocsátási jog van online vásárlásnál?",
        "expected_keywords": ["14 nap", "14", "elbocsátás", "visszavétel", "online"],
        "category": "fogyasztoverdelem"
    },
    {
        "id": 15,
        "question": "Mi a fogyasztóvédelmi hatóság neve Magyarországon?",
        "expected_keywords": ["OGYÉI", "Országos Fogyasztóvédelmi", "hatóság", "fogyasztó"],
        "category": "fogyasztoverdelem"
    },
    {
        "id": 16,
        "question": "Mi a fogyasztó legfontosabb joga a termékhiba esetén?",
        "expected_keywords": ["garancia", "javítás", "csere", "termékbírál", "visszatérítés"],
        "category": "fogyasztoverdelem"
    },
    
    # === Munkaerő-piac (2 kérdés) ===
    {
        "id": 17,
        "question": "Milyen támogatás érhető el új munkahely létrehozásakor?",
        "expected_keywords": ["munkahelyteremtési", "támogatás", "200", "500", "egyesztett"],
        "category": "munkaeropiac"
    },
    {
        "id": 18,
        "question": "Mikor jár nyugdíjba Magyarországon?",
        "expected_keywords": ["62 év", "62", "nyugdíj", "kor", "idős"],
        "category": "munkaeropiac"
    },
    
    # === Általános tudásbázis (2 kérdés) ===
    {
        "id": 19,
        "question": "Mennyi a munkavállalói egészségbiztosítási járulék mértéke?",
        "expected_keywords": ["18.5", "18,5", "járulé", "egészségbiztosít", "munkavállalói"],
        "category": "munkajog"
    },
    {
        "id": 20,
        "question": "Mire használható a GDPR által adható jog?",
        "expected_keywords": ["átvihetőség", "átvitel", "adat", "jog", "tiltakozás"],
        "category": "adoavedelem"
    },
]


def get_question_by_category(category: str) -> list:
    """Visszaad egy kategória kérdéseit."""
    return [q for q in EVALUATION_QUESTIONS if q["category"] == category]


def get_all_questions() -> list:
    """Visszaadja az összes kérdést."""
    return EVALUATION_QUESTIONS


def get_questions_by_count(count: int = 20) -> list:
    """Visszaad egy meghatározott számú kérdést."""
    return EVALUATION_QUESTIONS[:count]


if __name__ == "__main__":
    print(f"Összes kérdés: {len(EVALUATION_QUESTIONS)}")
    for cat in set(q["category"] for q in EVALUATION_QUESTIONS):
        count = len(get_question_by_category(cat))
        print(f"  {cat}: {count} kérdés")
