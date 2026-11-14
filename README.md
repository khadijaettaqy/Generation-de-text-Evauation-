# Évaluation des modèles NLP — TP pratique

Notebook pédagogique en Python consacré à l'évaluation de trois tâches de **traitement automatique du langage (NLP)** : **résumé automatique**, **question-réponse (QA/RAG)** et **génération de texte**. Une dernière section présente l'**évaluation humaine**.

Le projet permet de comprendre ce que chaque métrique mesure, de suivre ses calculs sur de petits exemples en français et de visualiser certains scores. Il ne constitue pas une plateforme d'évaluation de production.

## Objectifs

- Calculer et interpréter des métriques adaptées à chaque tâche NLP.
- Comparer des métriques complémentaires et comprendre leurs limites.
- Observer les effets des reformulations, des répétitions et des réponses partielles.
- Distinguer correspondance lexicale, similarité sémantique, fidélité aux sources et qualité humaine.

## Organisation des fichiers

```text
.
├── README.md
└── TP_Evaluation_NLP_3_Taches_Execute.ipynb
```

- `TP_Evaluation_NLP_3_Taches_Execute.ipynb` : version comportant des sorties de démonstration déjà enregistrées.
- `README.md` : documentation du code et des instructions d'exécution.



## Partie 1 — Résumé automatique (*Text Summarization*)

**Objectif :** comparer un résumé candidat à un résumé de référence.

| Métrique | Fonction / bibliothèque | Principe |
|---|---|---|
| ROUGE-1 | `rouge_n(reference, candidate, 1)` | Recouvrement des unigrammes |
| ROUGE-2 | `rouge_n(reference, candidate, 2)` | Recouvrement des bigrammes |
| ROUGE-L | `rouge_l(reference, candidate)` | Plus longue sous-séquence commune (LCS) |
| BLEU-2 | `bleu_n(candidate, references, 2)` | Précision modifiée des n-grammes et pénalité de longueur |
| BERTScore | `bert_score(...)`, optionnel | Correspondances entre embeddings contextualisés |

Exemple de données :

```python
resume_reference = "Le petit chat noir mange une souris dans le jardin"
resume_candidat = "Le chat mange une souris dans le jardin"

print(rouge_n(resume_reference, resume_candidat, 1))
print(rouge_n(resume_reference, resume_candidat, 2))
print(rouge_l(resume_reference, resume_candidat))
print(bleu_n(resume_candidat, [resume_reference], 2))
```

La fonction `rouge_n` renvoie notamment `precision`, `recall`, `f1` et les nombres de n-grammes. `rouge_l` renvoie aussi la longueur de la LCS. `bleu_n` renvoie le score BLEU et un dictionnaire contenant `BP`, `p1` et `p2` pour BLEU-2.

**Limite :** les calculs ROUGE/BLEU écrits dans le notebook sont des implémentations pédagogiques. BLEU est calculé **sans lissage**, si bien qu'un ordre de n-grammes sans correspondance rend le score nul. Les résultats ne reproduisent pas nécessairement tous les réglages des bibliothèques officielles.

## Partie 2 — Question Answering (QA) et RAG

**Objectif :** évaluer la correspondance d'une réponse avec une référence et son éventuel soutien par des documents.

| Métrique | Fonction / bibliothèque | Principe |
|---|---|---|
| Exact Match (EM) | `exact_match(reference, prediction)` | 1 si les réponses normalisées sont identiques, sinon 0 |
| Token F1 | `token_f1(reference, prediction)` | F1 du recouvrement des occurrences de mots |
| Similarité sémantique | `SentenceTransformer`, optionnel | Cosinus entre embeddings de réponses |
| Faithfulness | Calcul manuel sur `claims` | Proportion d'affirmations annotées comme étayées par le contexte |

Exemple :

```python
print(exact_match("Victor Hugo", "Hugo"))  # 0
print(token_f1("Victor Hugo", "Hugo"))      # environ 0.667
```

Le notebook contient un petit ensemble de questions portant notamment sur les capitales et des réponses courtes. EM et Token F1 sont calculés pour chaque question, puis agrégés en moyenne.

La normalisation utilisée dans le TP est **simplifiée pour le français** (minuscules, retrait de ponctuation, découpage en mots). Elle n'est pas identique à l'évaluation officielle SQuAD.

### Faithfulness : exemple manuel

Le contexte fictif décrit une offre Fibre Premium. Une liste `claims` regroupe quatre affirmations, chacune annotée avec :

- `claim` : l'affirmation à vérifier ;
- `supported` : booléen indiquant si elle est étayée ;
- `evidence` : justification textuelle.

Le score est calculé par :

```python
faithfulness = sum(x['supported'] for x in claims) / len(claims)
```

Dans l'exemple, **3 affirmations sur 4** sont étayées, donc `faithfulness = 0.75`. La vérification est **manuelle et illustrative** : le notebook ne met pas en place un juge LLM ou NLI automatisé.

## Partie 3 — Génération de texte (*Text Generation*)

| Métrique | Fonction / bibliothèque | Principe |
|---|---|---|
| Perplexity (PPL) | `perplexity(probs)` | Exponentielle de la cross-entropy moyenne |
| Distinct-1 | `distinct_n(texts, 1)` | Proportion d'unigrammes uniques |
| Distinct-2 | `distinct_n(texts, 2)` | Proportion de bigrammes uniques |
| Self-BLEU-2 | `self_bleu(texts, 2)` | BLEU de chaque texte contre les autres, puis moyenne |
| MAUVE | `mauve.compute_mauve(...)`, optionnel | Proximité statistique des distributions de textes humains et générés |

### Perplexity

Le notebook utilise les **probabilités fictives des tokens réellement observés** et non les sorties d'un vrai modèle entraîné :

```python
probabilities = [0.5, 0.25, 0.125]
cross_entropy, ppl = perplexity(probabilities)
print(cross_entropy, ppl)  # environ 1.3863, 4.0
```

Formule : `PPL = exp(-mean(log(p_i)))`. À corpus et tokenisation comparables, **une PPL plus faible** indique de meilleures probabilités attribuées aux tokens observés. Elle ne mesure pas directement la factualité ou l'utilité du texte.

### Diversité lexicale

```python
generations = [
    "Le ciel est bleu et calme",
    "Le ciel est bleu et lumineux",
    "Une forêt verdoyante entoure le lac",
    "Le soleil éclaire doucement les montagnes",
]

print(distinct_n(generations, 1))
print(distinct_n(generations, 2))
print(self_bleu(generations, 2))
```

`distinct_n` agrège les n-grammes sur le corpus **sans construire de bigrammes entre deux textes**. Self-BLEU utilise le BLEU pédagogique sans lissage du notebook et renvoie les scores individuels ainsi que leur moyenne.

En général : **Distinct-n élevé** = davantage de n-grammes différents ; **Self-BLEU faible** = moins de recouvrement lexical entre générations. Aucun de ces scores ne garantit la cohérence.

### MAUVE : précautions

L'exemple contient seulement quatre textes humains et quatre générations. Il sert à montrer les entrées attendues, **pas** à fournir un résultat MAUVE scientifiquement valide. Le code exige au moins 100 textes de chaque catégorie pour démarrer l'expérience, et recommande un ensemble beaucoup plus grand pour une analyse robuste. Les textes doivent être comparables en domaine et en longueur.

## Bonus — Évaluation humaine

Le notebook propose une grille de notation de **1 à 5** sur quatre dimensions :

| Dimension | Question évaluée |
|---|---|
| Fluency (fluidité) | Le texte est-il naturel et grammaticalement correct ? |
| Coherence (cohérence) | Les idées s'enchaînent-elles logiquement ? |
| Relevance (pertinence) | La réponse correspond-elle à la demande ? |
| Factuality (exactitude factuelle) | Les faits sont-ils exacts ? |

Deux exemples annotés sont fournis dans la liste `human_evaluation`. Le code affiche les notes individuelles et leur moyenne. Une moyenne élevée peut masquer une erreur factuelle grave : il est préférable de conserver les dimensions séparées.

## Fonctions utilitaires

| Fonction | Description |
|---|---|
| `tokens(text)` | Découpe un texte en mots, en minuscules et sans ponctuation |
| `ngrams(seq, n)` | Construit les n-grammes d'une séquence |
| `clipped_overlap(candidate_items, reference_items)` | Compte les occurrences communes avec plafonnement |
| `prf(overlap, candidate_count, reference_count)` | Calcule précision, rappel et F1 |
| `lcs_length(a, b)` | Calcule la longueur de la plus longue sous-séquence commune |
| `show_scores(scores)` | Affiche les métriques sous forme lisible |

## Visualisations et exercices

Le notebook produit notamment :

- un graphique des scores de résumé (ROUGE-1, ROUGE-2, ROUGE-L, BLEU-2) ;
- un graphique comparant Distinct-1, Distinct-2 et Self-BLEU-2 pour des générations variées et répétitives ;
- trois exercices guidés invitant à modifier des résumés, des réponses de QA, des probabilités et des générations.

Les graphiques sont réalisés avec `matplotlib`.

## Limites et reproductibilité

1. Les petits corpus et les valeurs de probabilités sont **des exemples pédagogiques**, pas des benchmarks de modèles entraînés.
2. Les métriques lexicales ne garantissent ni la compréhension du sens ni l'exactitude factuelle.
3. BERTScore et Sentence-BERT nécessitent des téléchargements et peuvent produire des résultats différents selon la version ou le modèle installé.
4. La métrique MAUVE nécessite davantage de textes et un protocole expérimental contrôlé.
5. Le score Faithfulness de l'exemple est basé sur des annotations **définies dans le code**, pas sur une vérification automatique.
6. Pour comparer plusieurs modèles, fixer le corpus, la tokenisation, les paramètres et la méthode d'agrégation.

## Résultats attendus et travail à rendre

Les cellules de base affichent les valeurs des métriques et les graphiques associés. Pour construire un court compte rendu :

1. Présenter une capture des résultats de chacune des trois parties.
2. Montrer deux exemples où des métriques complémentaires conduisent à des interprétations différentes.
3. Produire un tableau comparatif des métriques.
4. Expliquer pourquoi aucune métrique isolée ne garantit la qualité globale d'un système NLP.

## Références utiles

- [Hugging Face Evaluate — ROUGE](https://github.com/huggingface/evaluate/blob/main/metrics/rouge/README.md)
- [BERTScore — dépôt officiel](https://github.com/Tiiiger/bert_score)
- [Sentence Transformers — documentation](https://www.sbert.net/)
- [MAUVE — documentation](https://krishnap25.github.io/mauve/)

---

**Contexte :** support pratique en français pour une présentation pédagogique sur l'évaluation des modèles NLP.
