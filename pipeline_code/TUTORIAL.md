# Tutorial: Aufnahme & Training

Von der SplatKing-Aufnahme zum trainierten `.ply` / `.spz`. Praktischer Leitfaden mit
Empfehlungen für **Objekt** vs. **ganzen Raum**.

---

## 0. Umgebung einrichten

Im Projektordner `pipeline_code/` (siehe `README.md` → "Einrichtung" für die volle
Torch-/CUDA-/gsplat-Installation). Danach immer über `uv run` aufrufen:

```powershell
uv sync                              # leichte Tools (numpy/pillow/tqdm)
uv run python pipeline\run_pipeline.py --help
```

> `uv run` benutzt automatisch die in `pyproject.toml`/`uv.lock` gepinnten Versionen
> (u. a. NumPy `<2`, wichtig für den pycolmap-Reader) – kein manuelles venv-Aktivieren
> nötig.

---

## 1. Aufnahme mit SplatKing

### Modus wählen
- **LiDAR-Capture (empfohlen, iPhone Pro):** liefert direkt ein COLMAP-Modell mit
  **metrischen, schwerkraft-ausgerichteten Posen** + dichte LiDAR-Punktwolke. Bestes,
  stabilstes Ergebnis, Modell steht im Viewer aufrecht. → Ordner `LidarSeries_…`.
- **Foto-Capture (`photo_dual`):** nur Bilder, **keine Posen** → COLMAP muss die Posen
  erst rechnen (langsamer, scheitert eher bei wenig Überlappung). → Ordner `PhotoSeries_…`.

### Goldene Regeln (für beides)
- **Scharf bleiben:** langsam und ruhig bewegen, keine Bewegungsunschärfe.
- **Überlappung:** ~60–80 % zwischen aufeinanderfolgenden Bildern.
- **Gleichmäßiges Licht**, keine harten Wechsel; nichts Bewegtes im Bild.
- **Textur hilft:** matte, strukturierte Oberflächen rekonstruieren besser als glänzende,
  spiegelnde oder einfarbige.

### 🏺 Objekt (z. B. Vase)
- Das Objekt **mittig** halten und **360° umrunden** – nicht vom Fleck aus schwenken.
- **2–3 Ringe in verschiedenen Höhen** (Augenhöhe, tiefer, leicht von oben), auch
  Ober-/Unterseite mitnehmen.
- Objekt soll **einen guten Teil des Bildes füllen**; ~0,3–0,8 m Abstand.
- Grob **60–150 Bilder**.

### 🏠 Raum / Innenraum
- **Langsam abschreiten**, dabei die Kamera leicht schwenken, alle **Wände, Ecken,
  Übergänge** abdecken.
- **Schleifen schließen** (am Ausgangspunkt wieder vorbeikommen) – verbessert die
  Posenkonsistenz stark.
- Nicht zu schnell; in Türen/Engstellen extra langsam.
- Deutlich mehr Bilder als beim Objekt (**150–400+**).

---

## 2. Training starten (CLI)

Der Export liegt am besten in einem eigenen Ordner (z. B. `captures\…`), **nicht** in
`data\<scene>` (sonst Namenskollision).

### LiDAR-Capture (hat schon Posen)
```powershell
uv run python pipeline\run_pipeline.py --scene meinobjekt --input "C:\captures\LidarSeries_2026..."
```
Die Pipeline findet das `COLMAP_Text_Model` im Export automatisch (kein `--colmap` nötig).

### Foto-Capture (ohne Posen → COLMAP zuerst)
```powershell
uv run python pipeline\run_pipeline.py --scene meinobjekt --input "C:\captures\PhotoSeries_2026..." --colmap --drop-bands poor
```

**Ergebnis:** `results\meinobjekt\meinobjekt.ply` (+ `.spz`). Fortschritt läuft live im
Terminal; ~20–30 Min. Mit `--dry-run` nur den Befehl anzeigen ohne zu trainieren.

---

## 3. Parameter: Objekt vs. Raum

| Parameter | 🏺 Objekt | 🏠 Raum |
| --- | --- | --- |
| `--data-factor` | `1` (volle Auflösung) | `2` (halbe – spart VRAM) |
| `--max-steps` | `30000` | `50000` (größere, komplexere Szene) |
| `--sh-degree` | `3` | `3` |
| `--app_opt` | – | ✓ (siehe unten) |

**Warum?**
- Beide trainieren mit gsplats `default`-Strategie (freie Densification). Eine `mcmc`-
  Strategie (gedeckelte Gaussian-Zahl) haben wir für Räume ausprobiert, das führte aber
  zu Problemen – daher nicht mehr Teil dieser Pipeline.
- **`--app_opt`** (Appearance Optimization) gleicht Belichtungs-/Farbunterschiede
  zwischen den Aufnahmen aus – bei 150–400+ Fotos eines Raums ändert sich das Licht
  stärker als beim kurzen Objekt-Rundgang.
- **`--sh-degree`** (Spherical-Harmonics-Grad) steuert, wie viele Kugelflächenfunktions-
  Koeffizienten pro Gaussian für blickwinkelabhängige Farbe (z. B. Reflexionen) genutzt
  werden. `0` = nur Grundfarbe, höher = mehr Richtungsdetail, aber mehr Speicher; `3` ist
  gsplats Maximum und für beide Szenentypen sinnvoll.

Beispiele:
```powershell
# Objekt, maximale Qualität
uv run python pipeline\run_pipeline.py --scene vase --input "C:\captures\LidarSeries_..." --data-factor 1

# Raum, mehr Schritte + Appearance-Optimierung
uv run python pipeline\run_pipeline.py --scene wohnzimmer --input "C:\captures\LidarSeries_..." --data-factor 2 --max-steps 50000 -- --app_opt
```

### VRAM-Stellschrauben (12 GB)
1. **`--data-factor 2`** (oder `4`) – größter Hebel gegen „out of memory".
2. **`--max-steps`** reduzieren – schnelleres, etwas gröberes Ergebnis.
3. **`--app_opt`** kostet zusätzliches VRAM, ist aber bei ungleichmäßig belichteten
   Räumen die bessere Wahl als eine gedeckelte Gaussian-Zahl.

### Weitere Flags
- `--no-spz` – keine `.spz`-Konvertierung.
- `--data-dir <pfad>` – statt `--input` direkt auf ein fertiges Dataset/COLMAP-Modell zeigen.
- `--result-dir <pfad>` – Ausgabeordner überschreiben (Standard `results\<scene>`).

---

## 4. Ergebnis ansehen & freistellen

- **Ansehen:** `…\.spz` (oder `.ply`) in [SuperSplat](https://supersplat.play.canvas.app/)
  (lokal im Browser) reinziehen – Orbit/Zoom, und Streu-Gaussians mit dem Lasso löschen.
- **Objekt isolieren:** in SuperSplat eine Box um das Objekt ziehen, Rest löschen, neu
  exportieren. (Bei einem LiDAR-Raumscan ist drumherum noch Raum-Geometrie – das ist normal.)

---

## 5. Wenn die Qualität nicht stimmt
- **„Floater-Suppe" aus neuen Blickwinkeln:** meist zu wenig Abdeckung/Überlappung →
  gründlicher umrunden (Objekt) bzw. Schleifen schließen (Raum).
- **Foto-Capture, viele Frames nicht registriert:** `--drop-bands poor`, langsamer/mehr
  Überlappung aufnehmen, oder LiDAR-Modus nutzen.
- Mehr Hilfe: Troubleshooting-Tabelle in der `README.md`.
```
