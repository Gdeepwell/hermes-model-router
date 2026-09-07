# Model Router Plugin Telepítése

## Gyors telepítés GitHub-ról

Ha a plugin GitHub-on van:

```bash
hermes plugins install <your-username>/hermes-model-router
```

## Lokális telepítés

1. Klónozd vagy másold a plugin könyvtárat:

```bash
cp -r /tmp/model-router-plugin ~/.hermes/plugins/model-router
```

2. Engedélyezd a plugin-t:

```bash
hermes plugins enable model-router
```

3. Indítsd újra a Hermes-t vagy használj `/reset` parancsot

## Konfiguráció

A plugin automatikusan betölti a `router_config.yaml` fájlt. Testreszabáshoz:

```bash
# Másold a konfigurációt a saját könyvtáradba
cp ~/.hermes/plugins/model-router/router_config.yaml ~/model-router-config.yaml

# Szerkeszd
nano ~/model-router-config.yaml
```

A plugin a `~/.hermes/plugins/model-router/router_config.yaml` fájlt olvassa be alapértelmezetten.

## Ellenőrzés

Telepítés után ellenőrizheted a plugin állapotát:

```bash
hermes plugins list
```

Azt kell látnod: `model-router` - enabled

## Dashboard

A live dashboard indítása:

```bash
python3 ~/.hermes/plugins/model-router/web_viewer.py
```

Ezután nyisd meg: http://localhost:8765

## Hibaelhárítás

Ha a plugin nem töltődik be:

1. Ellenőrizd, hogy a plugin engedélyezve van:
   ```bash
   hermes plugins list
   ```

2. Nézd meg a Hermes logokat:
   ```bash
   tail -f ~/.hermes/logs/hermes.log
   ```

3. Indítsd újra a Hermes-t teljesen

## Frissítés

Ha új verzió érhető el:

```bash
cd ~/.hermes/plugins/model-router
git pull
# vagy manuálisan másold az új fájlokat
```

Majd `/reset` parancs a Hermes-ben.
