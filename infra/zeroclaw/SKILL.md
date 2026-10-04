---
name: langgraph
description: >-
  Route les questions de support vers le pipeline LangGraph.
author: yosra
version: 0.4.0
---

LangGraph

Utilise cette compétence pour toute question client nécessitant une recherche dans la documentation ou une vérification de statut.

Appelle l'outil http_request avec method=POST vers http://localhost:8001/webhook. Le body doit contenir: session_id, message (le message du client), channel, language, user_metadata (avec user_id et thread_id), et history (liste des messages précédents).

Réponds toujours avec le champ response de la réponse reçue.
