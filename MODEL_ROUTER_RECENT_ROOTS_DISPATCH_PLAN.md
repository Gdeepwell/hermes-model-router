# Model Router „legutóbbi root promptok” — dispatch terv

## Módosítás előtti bizonyíték
- Élő felület: `http://127.0.0.1:8765/` (helyi, futó `web_viewer.py`).
- Vizuálisan ellenőrizve: az „Utolsó rekordok” választó jelenleg `50 / 100 / 200 / 500 / Összes`, alapértéke `200`.
- A státusz nyers rekorddarabszámot jelez (`200 rekord`), miközben a látható lista root promptokat csoportosít és child/lifecycle sorokat rejt vagy a végrehajtási fához kapcsol.
- A projektben nem található tárolt PNG/JPG UI-kép; az élő felület volt az elérhető aktuális UI-bizonyíték.

## Célkontraktus
1. A választó valódi, látható root promptok számát korlátozza, ne a betöltött nyers router-sorok számát.
2. Opciók: `5 / 20 / 50`, alapérték: `20`; nyers `50 / 100 / 200 / 500` és `Összes` opció ne maradjon.
3. A backend elegendő előzményt olvasson ahhoz, hogy >200 frissebb child/lifecycle rekord mögött álló root promptok is megjelenjenek.
4. A modell-, keresés-, grouped/raw-, lifecycle- és worker-routing viselkedés maradjon működőképes.
5. Nincs Claude/Opus hívás, fallback vagy retry; nincs deploy/production/credential/destruktív művelet.

## Dispatch
### [sol] UI/vizuális elemzés + implementáció
- Szemlélje az élő UI-t és a jelenlegi forrást/teszteket.
- Valósítsa meg a 5/20/50 root-prompt limitet a UI/backend határon a legkisebb koherens változtatással.
- Adjon pontos fájl- és tesztbizonyítékot; ne deployoljon és ne indítsa újra a közös futó szolgáltatást.

### [spark] szűk read-only forrás-/teszt-felderítés
- Térképezze fel az endpoint, JS render/filter/group sorrendjét és a legjobb regressziós teszthelyet a >200 child/lifecycle esethez.
- Ne írjon fájlt és ne indítson modellt/fallbacket.

## Terra integráció és elfogadás
1. Worker-bizonyítékok ellenőrzése; mindegyikhez `SUPERVISOR DECISION: ACCEPT` vagy `REJECT`.
2. Az elfogadott Sol-változás integrációs felülvizsgálata.
3. Terra írja/egészíti ki a >200 child/lifecycle regressziós tesztet.
4. Fókuszált és teljes releváns tesztek + `py_compile` + whitespace/diff ellenőrzés.
5. A helyi viewer kontrollált újraindítása csak a végső élő verifikációhoz; screenshoton ellenőrizni a 5/20/50 opciókat és hogy a root promptlista nem üres.

## Supervisor decisions

### [sol] worker
**SUPERVISOR DECISION: ACCEPT.**

Indok: a worker az élő UI és a forrás vizsgálata után koherens UI/backend változást adott át; a választó 5/20/50, default 20, a kliens a root futásokat limitálja, a backend pedig 10 000 nyers rekordos kapcsolati előzményt szolgáltat. A megadott fókuszált tesztek zöldek voltak, a módosítást Terra újraolvasta és saját regressziós teszttel erősítette. Nem történt deploy vagy közös viewer-újraindítás.

### [spark] worker
**SUPERVISOR DECISION: ACCEPT.**

Indok: read-only feltárással pontosan azonosította a hibaseamet (`entries[-200:]` a kliensoldali `visibleEntries` előtt), a releváns függvénysorrendet és a >200 lifecycle regresszió fixture-jét. Nem módosított fájlt, nem indított szolgáltatást és nem használt Claude/Opus hidat.
