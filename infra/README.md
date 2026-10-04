# Infrastructure (Yosra)

Ce dossier contient la configuration infra côté ZeroClaw et stockage.

## zeroclaw/SKILL.md
Skill LangGraph enregistrée dans ZeroClaw. Permet à l'agent d'appeler le webhook LangGraph via `http_request` (POST, avec le schéma JSON complet : session_id, message, channel, language, user_metadata, history).

## database/schema.sql
Schéma PostgreSQL (base `felcloud_support`) : users, conversations, messages, tickets, agent_metadata.

## Infra déployée sur la VM (llm-inference-node-pulumi-1)
- Qdrant (port 6333) — collection `felcloud_support_kb`, 1024 dim, Cosine
- PostgreSQL (port 5432) — base `felcloud_support`
- Redis (port 6379) — mémoire de session
- ZeroClaw — agent `main`, webhook natif avec HMAC-SHA256 sur le port 8002
- Ollama sécurisé (accès local uniquement, plus d'exposition publique)
