# Model Router Plugin

Intelligens LLM routing Hermes Agenthez. Automatikus modellválasztás a feladat típusa alapján: Luna (egyszerű), Spark (read-only kód), Terra (default orchestrator), Sol (komplex/sensitive), és Claude Opus 5 bridge.

## Telepítés

### GitHub-ról (ajánlott)

```bash
hermes plugins install <your-username>/hermes-model-router
```

### Lokális telepítés

```bash
cd /path/to/model-router-plugin
hermes plugins install .
```

### Manuális telepítés

```bash
cp -r model-router ~/.hermes/plugins/
hermes plugins enable model-router
```

## Funkciók

### Automatikus Routing

- **Luna**: Egyszerű feladatok, rövid válaszok, alapvető kérdések
- **Spark**: Read-only kód elemzés, szűrt kódolási feladatok
- **Terra**: Alapértelmezett orchestrator, általános feladatok
- **Sol**: Komplex, biztonsági, kritikus feladatok
- **Claude Opus 5**: Standalone diagnostic bridge (opcionális)

### Privacy-First Audit Logging

- 240 karakteres bounded preview
- Redaktált sensitive adatok
- Parent/child correlation ID-k
- Token és költség metrikák

### Bounded Delegation

- Max 2 egyidejű child agent
- Max 1 spawn depth (flat hierarchy)
- Max 16 child iteráció
- Handoff capsule minden worker-nek

### Live Dashboard

```bash
python3 ~/.hermes/plugins/model-router/web_viewer.py
```

Megnyitja a `http://localhost:8765` címen a live dashboardot.

## Konfiguráció

A plugin automatikusan betölti a `router_config.yaml` fájlt. Testreszabás:

```bash
cp ~/.hermes/plugins/model-router/router_config.yaml ~/.hermes/model-router-config.yaml
```

Szerkeszd a `~/.hermes/model-router-config.yaml` fájlt a saját igényeid szerint.

### Fő beállítások

```yaml
# Modell elérhetőség
callable:
  luna: true
  spark: false  # read-only, explicit [spark] label szükséges
  terra: true
  sol: true
  opus5: true

# Default parent modell
default_model: terra

# Delegation korlátok
orchestration:
  enabled: false  # automatikus fan-out kikapcsolva
  max_tasks: 1

# Audit logging
logging:
  prompt_preview_chars: 240
  redact_prompt_preview: true
```

## Használat

### Explicit Modell Választás

```
[luna] Egyszerű kérdés
[spark] Olvasd el ezt a fájlt
[sol] Komplex biztonsági elemzés
[opus] Diagnostic review
```

### Delegáció

A parent agent automatikusan delegálhat független részfeladatokat:

```python
delegate_task(
  goal="Független részfeladat",
  context="Handoff capsule: cél, korlátok, releváns döntések"
)
```

## Policy

1. **Stabil Parent**: A user-facing beszélgetés nem vált modellt automatikusan
2. **Kivétel, nem szabály**: Delegáció csak valódi független feladatoknál
3. **Worker-korlátok**: Max 2 concurrent child, 1 spawn depth, 16 iteráció
4. **Privacy-safe audit**: 240 char bounded preview, redaktált sensitive adatok
5. **Dokumentáció**: Minden policy változás frissíti a README-t és teszteket

## Tesztek

```bash
cd ~/.hermes/plugins/model-router
pytest test_*.py
```

## Verzió

**1.2.0** - Stable parent policy, bounded delegation, privacy-safe logging

## License

MIT

## Szerző

SENTINEL - Hermes Agent Model Router
