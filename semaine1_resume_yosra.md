# Résumé Semaine 1 — Felcloud AI Support Agent

---

## Vue d'ensemble

**Felcloud AI Support Agent** est un agent d'assistance client intelligent. Il reçoit les messages clients via **ZeroClaw** (webhook), les analyse, cherche dans une base de connaissances, et génère une réponse. Si le client veut un humain, il crée automatiquement un ticket de support.

**Technologies principales :**

| Technologie | Rôle |
|---|---|
| **LangGraph** | Orchestration du workflow (graphe d'états) |
| **Qdrant** | Base vectorielle — stockage et recherche des connaissances |
| **LiteLLM** | Proxy LLM (compatible OpenAI, supporte LLaMA, Qwen...) |
| **FastAPI** | Serveur HTTP — reçoit les webhooks ZeroClaw |
| **PostgreSQL** | Persistance des conversations, messages, tickets |

---

## Ce qui a été fait (J1 à J7)

### J1 — Workflow LangGraph

L'état de chaque conversation est défini dans `GraphState` (Pydantic). Le graphe route les messages selon deux fonctions :

- `route_intent` → dirige vers `retrieval_node` (support), `generation_node` (chat), ou `escalation_node` (humain)
- `route_validation` → après génération, valide la réponse ou déclenche un retry

---

### J2 — Ingestion & Stockage des connaissances

Pipeline complet pour charger les documents dans Qdrant :

```
data/raw/ → loaders → normalize → chunk → embed (BGE-M3) → Qdrant
```

Types de documents supportés : **FAQ**, **TechDoc**, **Troubleshooting**, **Support Tickets**

**Configuration Qdrant :**
- Collection : `felcloud_support_kb`
- Vecteur dense : 1024 dimensions, distance Cosine
- Vecteur sparse : `bge_sparse` (recherche hybride)
- Index sur : `source_type`, `category`, `language`

**Schéma PostgreSQL** (`infra/database/schema.sql`) : tables `users`, `conversations`, `messages`, `tickets`, `agent_metadata` avec clés étrangères.

---

### J3 — Pipeline RAG (Retrieval-Augmented Generation)

Quand l'intent est `support`, l'agent cherche dans la base de connaissances :

1. Embedding de la question (dense + sparse via BGE-M3)
2. Recherche hybride dans Qdrant
3. Fusion des résultats par **RRF** (Reciprocal Rank Fusion)
4. Reranking (cross-encoder)
5. Assemblage du contexte + sources → envoyé au LLM pour générer la réponse

---

### J4 — Classification d'intention & Webhook ZeroClaw

**Webhook** (`/webhook/zeroclaw`) :
- Vérifie la signature **HMAC-SHA256** (header `X-ZeroClaw-Signature`) → 401 si invalide ou absente
- Valide le payload avec Pydantic → 400 si champs manquants
- Mappe le message vers `GraphState` et lance le graphe LangGraph

**Classifieur d'intention** (`intent_classifier.py`) :
- Utilise `ChatOpenAI` de LangChain — **pointé vers LiteLLM**, pas vers OpenAI
  > LiteLLM est un proxy compatible OpenAI. Ça permet d'utiliser LLaMA, Qwen, ou n'importe quel LLM avec la même interface. Le modèle est configuré via `STRUCTURED_OUTPUT_MODEL` dans `.env`.
- Sortie structurée (Pydantic) : `"support"` | `"escalation"` | `"general_chat"`
- Fallback automatique sur `"support"` si le LLM échoue

**Support multilingue (EN / FR / Darija) :**
Il n'y a pas de code spécifique par langue. Le LLM comprend nativement plusieurs langues.
Le champ `language` (`en` | `fr` | `darija`) est lu depuis le payload ZeroClaw et propagé dans `GraphState` pour que les autres nœuds (génération, contexte) puissent l'utiliser si nécessaire.

---

### J5 — Outils de monitoring & Génération de réponse

**4 outils LangChain** (dans `monitoring_tools`) :

| Outil | Rôle |
|---|---|
| `check_status(service_name)` | Statut d'un service (database, web_server...) |
| `get_resource_info(resource_id)` | Détails d'une VM ou ressource cloud |
| `get_metrics(service_name, metric_type)` | CPU, mémoire, latence |
| `get_logs(service_name, lines)` | Derniers logs d'un service |

Chaque outil fonctionne en **deux modes** selon la configuration `.env` :
- `ZEROCLAW_API_URL` renseigné → appel HTTP réel vers ZeroClaw API avec `Bearer {ZEROCLAW_API_KEY}`
- Vide → données simulées (mock) — pas de crash, utile en développement

**Nœud de génération** (`generation_node`) : même pattern LiteLLM, modèle défini par `GENERATION_MODEL` dans `.env`. Produit la réponse finale et attribue les sources si disponibles.

---

### J6 — Validation & Retry

Après chaque génération, la réponse passe par `validation_node` :

- Réponse vide ou < 10 caractères → `"failed"`
- Contient une phrase d'erreur (`"i don't know"`, `"as an ai"`, `"désolé je n'ai pas pu"`...) → `"failed"`
- Sinon → `"passed"`

Si `"failed"` et `retry_count < 2` → `retry_node` incrémente le compteur et renvoie vers la génération.
**Limite : 2 retries max** (`MAX_RETRIES = 2`) — 3 tentatives au total, puis END.

---

### J7 — Escalade vers un humain & Ticketing

> **L'escalade** se déclenche quand le client demande explicitement un humain (intent = `"escalation"`).
> L'agent n'essaie pas de répondre avec le LLM — il ouvre directement un ticket de support.

**`escalation_node`** :
- Génère une clé d'idempotence `uuid5(user_message)` — même message = même clé = pas de doublon
- Appelle `create_ticket` avec priorité `"high"`
- Réponse finale en français : *"J'ai créé le ticket TKT-XXXX, un agent humain vous contactera."*

**Outil `create_ticket`** :
- Si doublon (clé déjà utilisée) → retourne `"skipped"` immédiatement
- Si `TICKETING_API_URL` est renseigné → **POST HTTP vers l'API externe** avec `{user_id, issue_description, priority, idempotency_key}`
- Sinon → ticket fictif local `TKT-XXXXXXXX` (mode développement uniquement)

---

## Suite de tests — 150 tests ✅

| Fichier | Ce qui est testé |
|---|---|
| `test_j1_langgraph.py` | GraphState, routage, compilation du graphe, handoff ZeroClaw |
| `test_j2_ingestion.py` | Chunking, embeddings, schéma SQL |
| `test_j2_ingestion_data.py` | normalize.py, loaders.py, pipeline principal |
| `test_j3_rag.py` | RRF, reranking, contexte, dataset de régression golden |
| `test_j3_qdrant_client.py` | Connexion Qdrant, création collection, upsert |
| `test_j4_intent_webhooks.py` | Classification EN/FR/Darija, HMAC, payload tampered |
| `test_j5_monitoring_tools.py` | Schémas Pydantic, outils mock, enregistrement LangChain |
| `test_j5_generation_node.py` | Génération, attribution sources, gestion erreur LLM |
| `test_j6_validation_retry.py` | Validation, retry, boucle complète jusqu'à exhaustion |
| `test_j7_escalation.py` | Schéma ticket, idempotence, nœud d'escalade |

`tests/conftest.py` centralise les fixtures partagées : mock `langchain_openai`, factory `make_state`, `webhook_path`, `sign_payload`.

---

## Variables d'environnement

Tout passe par `app/config.py` (pydantic-settings → `.env`). **Aucun secret dans le code.**

### Variables obligatoires
```
LLM_BASE_URL              → URL du serveur LiteLLM
LLM_API_KEY               → Clé API LLM
QDRANT_URL                → URL de Qdrant
QDRANT_COLLECTION         → felcloud_support_kb
ZEROCLAW_WEBHOOK_SECRET   → Secret HMAC pour valider les webhooks
```

### Variables requises pour les intégrations live

> ⚠️ Sans ces variables, le code tourne en **mode mock** — acceptable en développement, **pas en production**.

```
ZEROCLAW_API_URL    → URL de l'API ZeroClaw (monitoring live)
ZEROCLAW_API_KEY    → Clé API ZeroClaw
TICKETING_API_URL   → URL de l'API de ticketing (tickets réels)
TICKETING_API_KEY   → Clé API ticketing
```

---

## Structure du projet

```
felcloud-ai-support-agent/
├── agent/server.py                  # Serveur FastAPI (point d'entrée)
├── app/
│   ├── config.py                    # Toutes les variables d'env
│   ├── graph/
│   │   ├── state.py                 # GraphState (Pydantic)
│   │   ├── router.py                # route_intent / route_validation
│   │   ├── workflow.py              # Compilation du graphe LangGraph
│   │   └── nodes/                   # intent_classifier, generation_node,
│   │                                #   validation_node, retry_node, escalation_node
│   ├── rag/
│   │   ├── ingestion/               # normalize, chunk, embed, loaders, main
│   │   └── retrieval/               # search (RRF, rerank), qdrant_client
│   ├── channels/webhooks/           # zeroclaw_webhook.py (HMAC + routing)
│   └── tools/                       # monitoring.py, ticketing.py
├── tests/
│   ├── conftest.py                  # Fixtures partagées
│   └── test_j*.py                   # Tests J1 à J7
├── infra/database/schema.sql        # Schéma PostgreSQL
├── data/evaluation/golden_dataset.json  # Dataset de régression RAG
└── .env.example                     # Template → copier en .env et remplir
```

---

## Lancer les tests

```bash
source venv/bin/activate

pytest tests/ -v        # tous les tests avec détail
pytest tests/ -q        # résumé rapide
pytest tests/test_j3_rag.py -v   # un fichier spécifique
```
