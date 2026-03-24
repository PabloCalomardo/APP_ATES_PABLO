# Changes

Data: 2026-02-18

## flow_core.py

- **Nova funció de mapping d'ID a bitmask (`uint64`)**
  - `source_id_to_bitmask(...)`: línia **38**

- **`split_release(...)` conserva IDs PRA i usa binari només per comptar/split**
  - Definició de funció: línia **104**
  - `release_binary = (release > 0).astype(np.uint8)`: línia **129**
  - `summ = np.sum(release_binary)`: línia **130**
  - Criteri de split amb `release_binary`: línia **142**

- **Propagació multi-label a `calculation(...)`**
  - Inicialització `source_id_array` (`np.uint64`): línia **195**
  - Càlcul de màscara de l'origen startcell: línia **226**
  - Acumulació multi-origen per cel·la (`bitwise_or`): línia **269**
  - Retorn ampliat amb `source_id_array`: línia **288**

- **Propagació multi-label a `calculation_effect(...)`**
  - Inicialització `source_id_array` (`np.uint64`): línia **323**
  - Càlcul de màscara de l'origen startcell: línia **354**
  - Acumulació multi-origen per cel·la (`bitwise_or`): línia **399**
  - Retorn ampliat amb `source_id_array`: línia **412**

## Simulation.py

- **Signal `finished` ampliada per transportar la capa d'orígens**
  - `finished = pyqtSignal(..., list)` amb 8 llistes: línia **38**

- **Recollida de resultats per procés**
  - Inicialització `source_id_list`: línia **103**
  - Assignació des de resultats (`res[5]`): línia **114**
  - Emissió final incloent `source_id_list`: línia **118**

## main.py

- **Flux GUI: nova capa global d'orígens i agregació multi-procés**
  - Inicialització `self.source_ids` (`np.uint64`): línia **342**
  - Signatura de `thread_finished(..., source_ids, ...)`: línia **353**
  - Fusió global OR bit-a-bit: línia **361**

- **Flux GUI: escriptura de sortida nova**
  - Escriptura `source_ids` amb `nodata=0`: línies **383-384**

- **Flux CLI: nova capa i agregació**
  - Inicialització `source_ids` (`np.uint64`): línia **525**
  - Inicialització `source_id_list`: línia **572**
  - Lectura per procés (`res[5]`): línia **581**
  - Fusió global OR bit-a-bit: línia **592**

- **Flux CLI: escriptura de sortida nova**
  - Escriptura `source_ids` amb `nodata=0`: línia **615**

## raster_io.py

- **Sortida raster amb `nodata` configurable**
  - Signatura: `output_raster(..., nodata=-9999)`: línia **56**
  - Aplicació a `AAIGrid`: línia **70**
  - Aplicació a `GTiff`: línia **72**

---

## Resultat funcional

- Es manté la simulació física existent.
- S'afegeix la traçabilitat multi-origen en una única execució.
- Es genera una nova capa `source_ids` on cada cel·la conté la combinació de PRAs contributors en format bitmask `uint64`.

---

## Update 2026-02-18 (Multibanda per PRA)

### Problema detectat

- El valor de la capa `source_ids` en bitmask (exemple: `3 = PRA1 + PRA2`) es podia interpretar com si fos un nou ID i semblava una sobrescriptura.

### Solució aplicada

#### main.py

- **Noves utilitats de decodificació i mapping**
  - `get_pra_ids(...)`: línia **45**
  - `build_source_multiband(...)`: línia **55**
  - `write_source_band_mapping(...)`: línia **68**

- **GUI**
  - Detecció d'IDs PRA des del raster d'entrada: línia **321**
  - Sortida bitmask renombrada a `source_ids_bitmask`: línia **416**
  - Nova sortida multibanda `source_ids_multiband.tif`: línia **423**
  - Export de mapping banda→PRA `source_ids_multiband_bands.csv`: línia **426**

- **CLI**
  - Detecció d'IDs PRA des del raster d'entrada: línia **540**
  - Sortida bitmask renombrada a `source_ids_bitmask`: línia **662**
  - Nova sortida multibanda `source_ids_multiband.tif`: línia **669**
  - Export de mapping banda→PRA `source_ids_multiband_bands.csv`: línia **672**

#### raster_io.py

- **Escriptura multibanda**
  - Detecció de raster 2D: línia **69**
  - Detecció de raster 3D (multi-band): línia **73**
  - Control d'error per `.asc` multibanda: línia **82**
  - Escriptura de totes les bandes (`new_dataset.write(raster)`): línia **94**

### Resultat

- Ara tens dues sortides:
  - `source_ids_bitmask.*` (compacta, per càlcul)
  - `source_ids_multiband.tif` (1 banda per PRA, sense ambigüitat visual)
- El fitxer `source_ids_multiband_bands.csv` indica quina banda correspon a cada PRA.
