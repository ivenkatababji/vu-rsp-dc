# Render deployment

This project can run on Render as a single Python web service:

- `https://<service>.onrender.com/` -> redirects to the web game
- `https://<service>.onrender.com/game` -> web client
- `https://<service>.onrender.com/admin` -> admin UI
- `https://<service>.onrender.com/...` -> API used by both the web client and the Expo mobile app

## What the service needs

- Python runtime
- `deployment/requirements.txt`
- a persistent disk mounted at `/var/data`
- `RPS_DB_PATH=/var/data/state.db`
- `RPS_INPUT_MODES=buttons,vision,audio`

The repo includes a Render blueprint at [`render.yaml`](C:\Users\Govin\Desktop\Mlops-Project\vu-rsp-dc\render.yaml).

## Authentication config

For production, prefer Render environment variables over committed JSON files:

- `RPS_ADMIN_USERNAME`
- `RPS_ADMIN_PASSWORD`
- `RPS_USERS_JSON`

Example `RPS_USERS_JSON`:

```json
{"guest":"guest","usr1":"pwd1","usr2":"pwd2","usr3":"pwd3","tiya":"rose"}
```

If these variables are omitted, the server falls back to the local config files under `deployment/src/server/`.

## Health check

Use:

```text
/healthz
```

## Mobile app production URL

Once the Render service URL exists, build the Expo app with:

```text
EXPO_PUBLIC_API_BASE_URL=https://<service>.onrender.com
```

When that variable is present, the mobile app automatically:

- uses the Render backend by default
- stops falling back to localhost-style URLs
- hides the manual API Origin input in production builds

## Render CLI notes

The installed CLI can validate the blueprint locally:

```bash
render blueprints validate
```

Creating or updating the actual Render service requires Render authentication on this machine.
