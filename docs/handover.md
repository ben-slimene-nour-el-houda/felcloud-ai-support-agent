# Handover — Felcloud AI Support Agent

Etat du projet a la fin du sprint 2 semaines (stage Yosra + Nour). Ce document donne une vue d'ensemble pour quiconque reprend le projet.

## Roles

- Yosra : ZeroClaw, infrastructure, securite, canaux
- Nour : LangGraph, RAG, outils, evaluation
- Sup Felcloud : responsable de stage, doc API ticketing et cas d'usage MQTT en attente

## Etat des taches (J1-J13)

- J1-J6 : fait (integration ZeroClaw-LangGraph, ingestion, RAG, classification, outils monitoring, validation/retry)
- J7 Ticketing : mock fonctionnel, PAS branche a un vrai systeme. Bloque en attente de la doc API "console" du sup
- J8 Email : fait, fonctionnel bout en bout
- J9 MQTT : composant actif mais non configure. Bloque en attente du cas d'usage metier du sup
- J10 Monitoring infra : fait (OOM Ollama diagnostique et corrige sur les 2 noeuds)
- J11 JWT/OAuth2 : fait (flux service-a-service ZeroClaw, voir section Securite)
- J12 Securite avancee : fait (faille RBAC monitoring corrigee)
- J13 Deploiement : en cours (ce document + runbook.md + systemd + secrets)

## Securite mise en place

- Webhook ZeroClaw : HMAC-SHA256 + anti-replay Redis (fenetre 300s)
- RBAC outils monitoring : role verifie a la construction du tool (make_monitoring_tools), plus expose au LLM comme parametre libre
- JWT service-a-service : ZeroClaw echange sa cle API (ZEROCLAW_API_KEY) contre un JWT court (60 min) via POST /auth/token. Le role est fixe a zeroclaw_agent, verifie par HMAC en amont
- caller_role transite maintenant dans GraphState, rempli au point d'entree webhook

## Ce qui N'EST PAS fait / limites connues

- Le flux JWT dashboard humain (login/password) n'existe pas. Seul le flux service-a-service ZeroClaw est implemente, car aucun dashboard n'existe encore dans le code
- make_monitoring_tools() n'est branche dans AUCUN noeud LangGraph pour le moment. Cote Nour : appeler make_monitoring_tools(state.caller_role) dans le noeud qui en a besoin
- TLS/HTTPS pas encore configure sur felcloud-agent (port 8001, HTTP simple actuellement). ZeroClaw (42617/42618) est deja en TLS
- Rate limiting Redis mentionne au plan (J11) mais pas implemente
- J7 (ticketing) et J9 (MQTT) restent bloques cote sup, voir section taches

## Documents lies

- docs/runbook.md : diagnostic et fix des incidents rencontres (Postgres, OOM, health check, gemma thinking)
- infra/database/schema.sql : schema PostgreSQL
- infra/zeroclaw/SKILL.md : config skill ZeroClaw-LangGraph
- .env.example : template a jour de toutes les variables d'environnement necessaires

## Prochaines etapes recommandees

1. Relancer le sup pour la doc API ticketing (J7) et le cas d'usage MQTT (J9), les deux bloquent depuis le debut
2. Nour : brancher make_monitoring_tools(state.caller_role) dans un noeud LangGraph des qu'un besoin de monitoring apparait dans le pipeline
3. TLS/HTTPS sur felcloud-agent avant tout deploiement expose publiquement
4. Decider si l'historique git doit etre nettoye (secrets historiques deja rotes, voir runbook.md section Gestion des secrets) ou laisse tel quel
5. Rate limiting Redis sur les endpoints exposes (mentionne au plan initial, jamais implemente)
