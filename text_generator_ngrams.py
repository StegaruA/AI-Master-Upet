"""
=============================================================
  GENERARE DE TEXT CU N-GRAMS
  Temă Master - Inteligență Artificială
=============================================================

DESCRIERE:
    Acest program implementează un model simplu de generare de text
    bazat pe tehnica N-grams. Modelul învață din texte de știri,
    construind o tabelă de probabilități pentru secvențe de cuvinte,
    apoi generează text nou prin eșantionare aleatorie.

CONCEPTE CHEIE:
    - N-gram: o secvență de N cuvinte consecutive dintr-un text
    - Bigramă (N=2): pereche de 2 cuvinte  → ("vremea", "este")
    - Trigramă (N=3): triplet de 3 cuvinte → ("vremea", "este", "frumoasă")
    - Model de limbaj: estimează probabilitatea unui cuvânt
      următor pe baza contextului (cuvintele anterioare)

FLUX:
    Text brut → Preprocesare → Construire N-grams → Generare text nou
"""

import random
import re
from collections import defaultdict, Counter


# =============================================================
# 1. DATE DE ANTRENAMENT (texte de știri simulate)
# =============================================================

# În practică, acestea ar fi încărcate dintr-un fișier sau API de știri.
# Folosim un corpus mic dar reprezentativ pentru domeniul știrilor.

CORPUS_STIRI = """
Guvernul a anunțat astăzi noi măsuri economice pentru susținerea întreprinderilor mici și mijlocii.
Pachetul de măsuri include reduceri de taxe și acces la credite cu dobânzi avantajoase.
Ministrul de finanțe a declarat că economia națională înregistrează o creștere semnificativă.
Creșterea economică este estimată la trei procente pentru acest an fiscal.
Banca Națională a menținut dobânda de referință la nivelul actual pentru a controla inflația.
Inflația a scăzut față de luna precedentă, potrivit datelor oficiale publicate ieri.

Echipa națională de fotbal a câștigat meciul de calificare cu un scor de două la unu.
Antrenorul principal a declarat că jucătorii au demonstrat o evoluție remarcabilă în teren.
Stadionul a fost plin la capacitate maximă cu peste cincizeci de mii de suporteri entuziaști.
Suporterii au aplaudat îndelung echipa la finalul meciului foarte disputat.
Federația de fotbal a anunțat că urmează meciuri importante în luna viitoare.
Jucătorii au mulțumit fanilor pentru susținerea extraordinară din tribune.

Cercetătorii au descoperit o nouă metodă de tratament pentru o boală rară.
Studiul clinic a implicat sute de pacienți din mai multe țări europene.
Rezultatele sunt promițătoare și vor fi publicate într-o revistă medicală internațională.
Medicii speră că noul tratament va fi disponibil publicului larg în doi ani.
Ministerul sănătății a alocat fonduri suplimentare pentru cercetare medicală avansată.
Spitalele din țară vor beneficia de echipamente moderne în urma investițiilor recente.

Tehnologia de inteligență artificială transformă industria și piața muncii globale.
Companiile investesc miliarde de dolari în dezvoltarea sistemelor inteligente de calcul.
Experții avertizează că unele locuri de muncă vor fi automatizate în deceniul următor.
Guvernele lumii discută reglementări pentru utilizarea responsabilă a inteligenței artificiale.
Universităților li se recomandă să introducă cursuri de programare și automatizare.
Studenții sunt încurajați să dezvolte competențe digitale pentru piața muncii viitoare.

Vremea se răcește considerabil în această săptămână în toată țara.
Meteorologii anunță ploi abundente și vânturi puternice pentru weekend.
Temperaturile vor scădea sub zero grade în zonele montane înalte.
Autoritățile recomandă cetățenilor să evite călătoriile pe timp de noapte.
Drumarii intervin cu utilaje pe șoselele acoperite de gheață și zăpadă.
Populația este sfătuită să se aprovizioneze cu alimente și medicamente necesare.

Parlamentul a votat o nouă lege privind protecția mediului înconjurător.
Legea prevede sancțiuni severe pentru companiile care poluează apa și aerul.
Organizațiile ecologiste au salutat adoptarea acestei legislații importante.
Industriașii cer o perioadă de tranziție mai lungă pentru adaptarea la noile norme.
Comisia Europeană monitorizează progresul statelor membre în privința protecției mediului.
România trebuie să atingă țintele de reducere a emisiilor până în anul doua mii treizeci.
"""


# =============================================================
# 2. PREPROCESARE TEXT
# =============================================================

def preproceseaza_text(text):
    """
    Curăță și tokenizează textul brut în listă de cuvinte.

    Pași:
        1. Convertire la litere mici (normalizare)
        2. Eliminarea caracterelor speciale nedorite
        3. Împărțirea în cuvinte individuale (tokenizare)
        4. Filtrarea cuvintelor prea scurte

    Args:
        text (str): Textul brut de intrare

    Returns:
        list: Listă de cuvinte procesate (tokens)
    """
    # Pas 1: Convertim totul la litere mici pentru uniformitate
    text = text.lower()

    # Pas 2: Înlocuim caracterele de punctuație cu spații
    # Păstrăm literele (inclusiv diacritice românești) și cifrele
    text = re.sub(r'[^\w\sșțăîâșțăîâ]', ' ', text)

    # Pas 3: Împărțim textul în cuvinte (tokenizare simplă)
    cuvinte = text.split()

    # Pas 4: Eliminăm cuvintele cu mai puțin de 2 caractere
    cuvinte = [c for c in cuvinte if len(c) >= 2]

    return cuvinte


# =============================================================
# 3. CONSTRUIREA MODELULUI N-GRAMS
# =============================================================

def construieste_ngrams(tokens, n=2):
    """
    Construiește un model N-gram din lista de tokens.

    Un model N-gram mapează fiecare secvență de (N-1) cuvinte
    la lista de cuvinte care o urmează în corpus.

    Exemplu pentru N=2 (bigrame):
        tokens = ["azi", "este", "o", "zi", "frumoasă"]
        model  = {
            ("azi",)    : ["este"],
            ("este",)   : ["o"],
            ("o",)      : ["zi"],
            ("zi",)     : ["frumoasă"]
        }

    Args:
        tokens (list): Lista de cuvinte din corpus
        n (int): Ordinul N-gramului (2=bigramă, 3=trigramă)

    Returns:
        dict: Modelul N-gram { context_tuple : [cuvinte_posibile] }
    """
    # defaultdict(list) creează automat o listă goală pentru chei noi
    model = defaultdict(list)

    # Parcurgem textul cu o fereastră glisantă de dimensiune N
    for i in range(len(tokens) - n):
        # Contextul = primele (N-1) cuvinte din fereastră
        context = tuple(tokens[i:i + n - 1])

        # Cuvântul următor = ultimul cuvânt din fereastră
        cuvant_urmator = tokens[i + n - 1]

        # Adăugăm cuvântul următor în lista asociată contextului
        model[context].append(cuvant_urmator)

    return model


def statistici_model(model, tokens):
    """
    Afișează statistici despre modelul construit.

    Args:
        model (dict): Modelul N-gram
        tokens (list): Lista de tokens din corpus
    """
    print("=" * 55)
    print("  STATISTICI MODEL")
    print("=" * 55)
    print(f"  Total cuvinte în corpus  : {len(tokens)}")
    print(f"  Vocabular unic           : {len(set(tokens))} cuvinte")
    print(f"  Contexte unice (N-grams) : {len(model)}")

    # Găsim cele mai frecvente contexte
    frecvente = sorted(model.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    print(f"\n  Top 5 contexte frecvente:")
    for context, urmatori in frecvente:
        print(f"    {' '.join(context):20s} → {len(urmatori)} continuări posibile")
    print("=" * 55)


# =============================================================
# 4. GENERAREA TEXTULUI NOU
# =============================================================

def genereaza_text(model, n, lungime=30, seed=None):
    """
    Generează text nou folosind modelul N-gram prin eșantionare.

    Algoritmul:
        1. Alegem un context de start (aleatoriu sau specificat)
        2. La fiecare pas, privim contextul curent
        3. Alegem aleatoriu un cuvânt din lista de continuări posibile
        4. Actualizăm contextul și repetăm până la lungimea dorită

    Această abordare se numește "Markov Chain" - cuvântul următor
    depinde DOAR de contextul imediat (nu de tot textul anterior).

    Args:
        model (dict): Modelul N-gram antrenat
        n (int): Ordinul N-gramului folosit la antrenament
        lungime (int): Numărul de cuvinte de generat
        seed (tuple): Context de pornire (opțional)

    Returns:
        str: Textul generat
    """
    # Dacă nu avem seed, alegem un context aleatoriu din model
    if seed is None:
        context_curent = random.choice(list(model.keys()))
    else:
        context_curent = seed

    # Inițializăm textul generat cu cuvintele din contextul de start
    text_generat = list(context_curent)

    # Generăm cuvintele unul câte unul
    for _ in range(lungime):
        # Căutăm continuările posibile pentru contextul curent
        continuari_posibile = model.get(context_curent)

        # Dacă nu există continuări (capăt de corpus), ne oprim
        if not continuari_posibile:
            break

        # Alegem aleatoriu un cuvânt următor din continuările posibile
        # Cuvintele care apar mai frecvent au șanse mai mari să fie alese
        # (pentru că apar de mai multe ori în lista de continuări)
        cuvant_nou = random.choice(continuari_posibile)

        # Adăugăm cuvântul la textul generat
        text_generat.append(cuvant_nou)

        # Actualizăm contextul: eliminăm primul cuvânt, adăugăm cel nou
        # Exemplu context (n=2): ("vremea", "este") → ("este", "frumoasă")
        context_curent = tuple(text_generat[-(n - 1):])

    return ' '.join(text_generat)


def genereaza_cu_tema(model, n, tema_cuvant, lungime=25):
    """
    Generează text pornind de la un cuvânt cheie specific.

    Caută în model toate contextele care conțin cuvântul temă
    și alege unul ca punct de pornire.

    Args:
        model (dict): Modelul N-gram
        n (int): Ordinul N-gramului
        tema_cuvant (str): Cuvântul cheie de pornire
        lungime (int): Lungimea textului de generat

    Returns:
        str: Textul generat sau mesaj de eroare
    """
    # Căutăm toate contextele care conțin cuvântul temă
    contexte_cu_tema = [
        ctx for ctx in model.keys()
        if tema_cuvant.lower() in ctx
    ]

    if not contexte_cu_tema:
        return f"[Cuvântul '{tema_cuvant}' nu a fost găsit în corpus]"

    # Alegem aleatoriu unul din contextele găsite
    context_start = random.choice(contexte_cu_tema)

    return genereaza_text(model, n, lungime=lungime, seed=context_start)


# =============================================================
# 5. EVALUARE: PERPLEXITATE
# =============================================================

def calculeaza_perplexitate(model, tokens_test, n):
    """
    Calculează perplexitatea modelului pe date de test.

    Perplexitatea măsoară "surpriza" modelului față de text nou.
    - Valoare mică = modelul prezice bine textul → mai bun
    - Valoare mare = modelul este "surprins" de text → mai slab

    Formulă: PP = 2^(-1/N * Σ log2(P(w|context)))

    Args:
        model (dict): Modelul N-gram
        tokens_test (list): Tokenuri de test
        n (int): Ordinul N-gramului

    Returns:
        float: Valoarea perplexității
    """
    import math

    log_prob_total = 0
    count = 0

    for i in range(len(tokens_test) - n):
        context = tuple(tokens_test[i:i + n - 1])
        cuvant_real = tokens_test[i + n - 1]

        continuari = model.get(context, [])

        if continuari:
            # Probabilitatea cuvântului real = frecvență relativă
            freq_cuvant = continuari.count(cuvant_real)
            probabilitate = freq_cuvant / len(continuari)

            if probabilitate > 0:
                log_prob_total += math.log2(probabilitate)
                count += 1

    if count == 0:
        return float('inf')

    perplexitate = 2 ** (-log_prob_total / count)
    return round(perplexitate, 2)


# =============================================================
# 6. PROGRAM PRINCIPAL
# =============================================================

def main():
    print("\n" + "=" * 55)
    print("  GENERARE TEXT CU N-GRAMS — MODEL DE LIMBAJ")
    print("  Temă Master Inteligență Artificială")
    print("=" * 55)

    # --- Preprocesare ---
    print("\n[1] Preprocesare corpus...")
    tokens = preproceseaza_text(CORPUS_STIRI)
    print(f"    Corpus procesat: {len(tokens)} tokens")

    # --- Antrenare modele cu N diferit ---
    print("\n[2] Antrenare modele N-gram...")

    model_bigram  = construieste_ngrams(tokens, n=2)
    model_trigram = construieste_ngrams(tokens, n=3)
    model_4gram   = construieste_ngrams(tokens, n=4)

    print(f"    Bigramă  (N=2): {len(model_bigram)} contexte")
    print(f"    Trigramă (N=3): {len(model_trigram)} contexte")
    print(f"    4-gram   (N=4): {len(model_4gram)} contexte")

    # --- Statistici ---
    print()
    statistici_model(model_trigram, tokens)

    # --- Generare text ---
    print("\n[3] Generare text nou...\n")

    random.seed(42)  # Seed fix pentru reproducibilitate

    # Comparație între diferite valori ale lui N
    for n, model, eticheta in [
        (2, model_bigram,  "BIGRAMĂ  (N=2) — context 1 cuvânt"),
        (3, model_trigram, "TRIGRAMĂ (N=3) — context 2 cuvinte"),
        (4, model_4gram,   "4-GRAM   (N=4) — context 3 cuvinte"),
    ]:
        print(f"  [{eticheta}]")
        text = genereaza_text(model, n=n, lungime=20)
        print(f"  {text}")
        print()

    # Generare cu temă specifică
    print("  [GENERARE PE TEME SPECIFICE — Trigramă]\n")
    for tema in ["economia", "fotbal", "tehnologia", "vremea"]:
        text = genereaza_cu_tema(model_trigram, n=3, tema_cuvant=tema, lungime=18)
        print(f"  Tema '{tema}':")
        print(f"  {text}\n")

    # --- Evaluare ---
    print("[4] Evaluare model (Perplexitate)...")
    # Folosim o parte din corpus ca date de test
    tokens_test = tokens[:100]
    for n, model, eticheta in [
        (2, model_bigram,  "Bigramă"),
        (3, model_trigram, "Trigramă"),
        (4, model_4gram,   "4-gram"),
    ]:
        pp = calculeaza_perplexitate(model, tokens_test, n)
        print(f"    {eticheta:12s}: perplexitate = {pp}")

    print("\n" + "=" * 55)
    print("  CONCLUZIE:")
    print("  - N mai mare = context mai lung = text mai coerent")
    print("  - N prea mare = model memorează corpus, nu generalizează")
    print("  - N optim depinde de dimensiunea corpusului")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
