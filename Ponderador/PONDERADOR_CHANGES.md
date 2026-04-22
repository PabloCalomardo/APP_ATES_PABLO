# Ponderador Change Log

Aquest document registra l'evolucio del classificador del directori `Ponderador`.
Objectiu: poder fer canvis estructurals sense perdre traçabilitat i poder comparar resultats abans/despres.

## Scope

- Modul principal: `Ponderador/AutoATES_classifier.py`
- Integracio: `main.py` (step 14 invoca `run_autoates_weighted`)

## Baseline (versio de partida abans dels canvis d'aquesta iteracio)

La versio de partida del ponderador (estat actual abans d'aplicar nous ajustos) funcionava aixi:

1. Entrades principals:
- DEM
- Canopy/forest
- Cell count / exposure (`Exposure_zdelta_cellcount.tif` a la pipeline)
- Flow-Py travel angle (`FP_travel_angle.tif`)
- PRA binari (`pra_basin_*.tif`)

2. Reclassificacions i combinacio:
- Reclassificacio de pendent (SAT01/SAT12/SAT23/SAT34)
- Reclassificacio de Flow-Py alpha angles (AAT1/AAT2/AAT3)
- Reclassificacio de cell count (CC1/CC2)
- Reclassificacio de forest segons tipus (`pcc`, `bav`, `stems`, `sen2cc`)
- Aplicacio de mascara PRA
- Taula de mapping a classes finals ATES (0..4)

3. Generalitzacio:
- Eliminacio de clumps petits (< ISL_SIZE)
- `fillnodata` per suavitzar discontinuïtats
- Sortida principal: `ates_gen.tif` i copia a `Ponderador_ATES.tif`

4. Limitacio detectada a verificacio:
- Biaix conservador (underprediction), especialment:
  - classe 3 predita com 2
  - classe 4 predita com 3
- Errors mes alts en zones de frenada (`Star_propagating_Ending_Zones`, valor 3)

---

## Canvis aplicats en aquesta iteracio (2026-04-22)

### Canvi 1: Post-ajust en zones de frenada (ending-zone boost)

**Objectiu**
- Reduir underprediction en classes 3/4 a zones on l'analisi ha mostrat mes error.

**Implementacio**
- Nou comportament dins `run_autoates_weighted` (despres de generar `ates_gen.tif`).
- Es construeix la unio de zones de frenada llegint:
  - `BasinX/Star_propagating_Ending_Zones/Ava_*.tif` (valor 3 = ending)
- Es calcula llindar d'exposicio per quantil sobre `cell_count_path` (per defecte q=0.60).
- Regla aplicada:
  - domini: pixels ending
  - condicio: exposicio >= quantil
  - promocio: 2 -> 3, 3 -> 4
- Es guarda informe d'aplicacio a:
  - `BasinX/ending_zone_boost_report.json`

**Paràmetres afegits a `run_autoates_weighted`**
- `ending_zone_boost=True`
- `ending_zone_exposure_quantile=0.60`

**Funcions noves**
- `_ending_zone_union_mask(...)`
- `_apply_ending_zone_boost(...)`

---

### Canvi 2: Ajust de llindars per reduir biaix conservador

**Objectiu**
- Relaxar lleugerament la frontera cap a classes altes per atacar errors 3->2 i 4->3.

**Llindars modificats**
- `SAT23`: 28 -> 27
- `SAT34`: 39 -> 38
- `AAT2`: 24 -> 23
- `AAT3`: 33 -> 32

**Nota**
- Son ajustos inicials (heuristics) per validar tendencia.
- Si millora parcialment, el seguent pas recomanat es calibratge sistematic (grid-search curt) sobre Verificador.

---

## Fitxers modificats en aquesta iteracio

1. `Ponderador/AutoATES_classifier.py`
- Ajust de llindars SAT/AAT
- Ending-zone boost amb informe JSON
- Nous parametres de `run_autoates_weighted`

2. `Ponderador/PONDERADOR_CHANGES.md`
- Aquest document de traçabilitat

---

## Compatibilitat i impacte

- No es trenca la invocacio existent de `main.py`.
- `step_14_ponderador_autoates` continua cridant `run_autoates_weighted` sense canviar signatura obligatoria.
- Els nous parametres tenen valors per defecte.
- S'espera canvi de resultats ATES en qualsevol nova execucio del step 14 (i per tant en la comparacio Verificador).

---

## Properes passes recomanades

1. Executar pipeline amb la nova versio del ponderador.
2. Reexecutar comparador de Verificador.
3. Revisar especialment:
- underprediction global
- F1 de classes 3 i 4
- metriques a ending zones
4. Si encara hi ha biaix, calibrar:
- `ending_zone_exposure_quantile`
- offsets SAT/AAT
- regles de promocio condicionades (ex: afegir runout/terrain proxies)

---

## Rollback d'estrategia (2026-04-22, tarda)

Per decisio de validacio, s'ha revertit el classificador a la versio inicial de ponderador abans dels dos ajustos anteriors.

### Estat actual actiu a codi

1. Llindars restaurats:
- `SAT23 = 28`
- `SAT34 = 39`
- `AAT2 = 24`
- `AAT3 = 33`

2. S'ha eliminat del flux actiu:
- post-ajust de zones de frenada (`ending-zone boost`)
- funcions auxiliars associades
- parametres extra a `run_autoates_weighted`
- `ending_zone_boost_report.json`

### Motivacio del rollback

- Els resultats d'EXP2 mostraven reduccio d'underprediction pero increment important d'overprediction, especialment a Bow Summit.
- Es prioritzara primer ajustar parametres de simulacio Flow-Py (fora de ponderador) i despres es tornara a introduir una logica de zones de frenada pero en sentit contrari (degradar classe a ending zones quan pertoqui).

### Notes

- Aquest document manté l'historial complet: els canvis no s'esborren del registre encara que el codi s'hagi revertit.
