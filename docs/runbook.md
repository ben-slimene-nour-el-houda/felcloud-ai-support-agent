# Runbook — Felcloud AI Support Agent

Ce document couvre les incidents reels rencontres en production/staging et comment les diagnostiquer et les resoudre.

## Vue d'ensemble des services

| Service | Type | Port | Commande de statut |
|---|---|---|---|
| Ollama (local) | systemd | 11434 | sudo systemctl status ollama |
| Ollama (distant, pulumi-2) | systemd | 11434 | ssh -i ~/.ssh/llm-key.pem ubuntu@192.168.100.80 |
| LiteLLM | systemd | 4000 | sudo systemctl status litellm |
| PostgreSQL | systemd (cluster) | 5432 | sudo pg_lsclusters |
| felcloud-agent | systemd | 8001 | sudo systemctl status felcloud-agent |
| ZeroClaw | podman | 42617 | sudo podman ps |
| Redis | podman | 6379 | redis-cli -a PASSWORD PING |
| Qdrant | podman | 6333-6334 | curl localhost:6333/collections |

## Incident : LiteLLM repond "service unavailable" ou timeout

Symptome : un appel via LiteLLM (:4000) echoue, alors qu'Ollama en direct (:11434/api/tags) repond normalement.

Cause n1 - PostgreSQL down. LiteLLM utilise Prisma pour se connecter a Postgres (spend tracking). Si Postgres est down, LiteLLM crash-loop toutes les ~5s. Log typique : "Can't reach database server at localhost:5432", error_code P1001.

Diagnostic :
sudo pg_lsclusters
sudo ss -tlnp | grep 5432

Fix :
sudo pg_ctlcluster 14 main start

Note : postgresql.service (parapluie) peut afficher "active (exited)" meme si le vrai cluster postgresql@14-main est down. Toujours verifier avec pg_lsclusters.

## Incident : Ollama OOM Killer sur un noeud

Symptome : erreur "Server disconnected" quand LiteLLM appelle un gros modele. Log : "A process of this unit has been killed by the OOM killer", "Failed with result oom-kill".

Cause : plusieurs modeles charges en meme temps depassent la RAM disponible (15Go sur ces VMs).

Fix (par noeud, chaque noeud Ollama doit avoir ce override) :
sudo mkdir -p /etc/systemd/system/ollama.service.d
echo 'Environment="OLLAMA_MAX_LOADED_MODELS=1"' | sudo tee -a /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama

Verifier apres coup avec free -h et en rechargeant un gros modele.

## Incident : /health de LiteLLM montre des faux "unhealthy" (408 timeout)

Symptome : GET /health montre certains modeles unhealthy avec "Timeout exceeded", alors que de vraies requetes passent normalement.

Cause : avec OLLAMA_MAX_LOADED_MODELS=1, les health checks paralleles sur plusieurs modeles font la queue pour charger, et certains depassent le timeout par defaut du health check.

Fix : ajouter dans config.yaml sous general_settings :
health_check_ignore_transient_errors: true

Ca n'empeche pas les 408 d'apparaitre dans /health, mais ca empeche ces faux positifs d'affecter le routing reel.

## Incident : un modele "thinking" (ex gemma4:12b) renvoie un content vide

Symptome : reponse API avec content: "" mais completion_tokens > 0.

Cause : le modele a un champ "thinking" separe (raisonnement interne). Avec un petit max_tokens, tout le budget est consomme par le thinking, rien ne reste pour le content.

Fix : ajouter think: false dans config.yaml pour ce modele, dans litellm_params, sur chaque entree (local + distants).

## Gestion des secrets

- .env ne doit JAMAIS etre commite. Verifier avec git status avant chaque commit.
- .env.example DOIT etre tenu a jour (c'est le seul fichier .env* qui est commite, voir .gitignore : !.env.example)
- Si une nouvelle variable est ajoutee dans .env, l'ajouter aussi dans .env.example avec une valeur placeholder
- Rotation : si un secret fuite (meme dans l'historique git), le roter immediatement cote service concerne puis mettre a jour .env
- Un secret peut fuiter dans l'historique git meme si .gitignore l'exclut aujourd'hui, s'il a ete commite avant que la regle existe. Verifier avec : git log --oneline --all -- .env

## Checklist de redemarrage complet (apres reboot VM ou panne generale)

1. sudo pg_lsclusters                    -> doit etre online, sinon sudo pg_ctlcluster 14 main start
2. sudo systemctl status ollama          -> sur chaque noeud (local + 192.168.100.80)
3. sudo systemctl status litellm         -> doit etre stable apres ~1 min (connexion Prisma)
4. sudo systemctl status felcloud-agent  -> API LangGraph sur :8001
5. sudo podman ps                        -> zeroclaw, redis, qdrant actifs
6. curl -s http://localhost:8001/auth/token -H "X-API-Key: ..." -> confirme que l'auth JWT fonctionne
