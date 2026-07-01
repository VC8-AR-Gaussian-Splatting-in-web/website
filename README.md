# Augmented Reality & 3D Gaussian Splatting im Web

Interaktive Quarto-Webseite zu **Thema 8 — Augmented Reality + Gaussian Splatting im Web**
(Modul Visual Computing). Sie führt vom theoretischen Hintergrund von 3D Gaussian Splatting
(3DGS) über die Capture-/Trainings-Pipeline bis zur interaktiven Darstellung im Browser:
ein Orbit-Viewer, eine begehbare Szene und eine WebXR-AR-Demo.

- **Live-Seite:** via GitHub Pages (Branch `gh-pages`, automatisch deployt — siehe
  [Veröffentlichen](#veröffentlichen)).
- **Technik:** [Quarto](https://quarto.org/) + [Babylon.js](https://www.babylonjs.com/)
  (WebGL/WebGPU, nativer Gaussian-Splatting-Support), WebXR für AR.

---

## Projektstruktur

| Seite / Datei | Owner | Inhalt |
|---|---|---|
| `index.qmd` | C | Startseite, Überblick, Team |
| `theorie/index.qmd` | A | Geschichte (NeRF→3DGS), Repräsentation, Rendering, Vergleich |
| `pipeline/index.qmd` | B | Use-Cases, Aufnahme, COLMAP, LichtFeld-Training, Export |
| `web-ar/index.qmd` | C | Babylon.js, WebXR, Orbit-Viewer, AR-Demo, Begehung („X") |
| `demo/index.qmd` | C | Interaktives Tutorial: eigenen Splat-Viewer selbst einbinden |
Zentrale Dateien:

- **`_quarto.yml`** — Projektkonfiguration: Navbar, Theme (`cosmo`), `freeze: auto`,
  Render-Globs.
- **`references.bib`** — Literaturverweise (Quarto-Zitate).
- **`styles.css`** — projektweite CSS-Ergänzungen.
- **`.github/workflows/publish.yml`** — CI: rendert & deployt nach gh-pages.

> Wer macht was: **A** = Sven Fydrich = Theorie & Methodik, **B** = Dominik Pleimes = Capture & Pipeline (eigenes 3DGS-Modell),
> **C** = Friedrich Commichau = Web & AR Integration.

### `pipeline_code/` — die Trainings-Pipeline hinter der Capture-Seite

Python-Projekt: **`pipeline_code/`** (eigenes `uv`-Environment, eigene `pyproject.toml`).
Es ist **nicht** Teil des Quarto-Builds und wird von der CI **nicht** ausgeführt — die
Website zeigt nur die fertigen Ergebnisse (Bilder/Videos in `pipeline/img|video/`, die
`.spz`-Modelle in `web-ar/models/`). `pipeline_code/` verwandelt eigene Handy-/LiDAR-
Aufnahmen in genau diese `.ply`/`.spz`-Modelle, braucht dafür aber **Windows + eine
NVIDIA-GPU** (CUDA) und ist entsprechend eigenständig dokumentiert:
[`pipeline_code/README.md`](pipeline_code/README.md) (volle Einrichtung, alle
Pipeline-Stufen, Fehlersuche) und [`pipeline_code/TUTORIAL.md`](pipeline_code/TUTORIAL.md)
(kürzerer, praktischer Leitfaden). Ein kleiner Beispiel-Datensatz (`pipeline_code/data/vase/`)
liegt mit im Repo, damit sich zumindest das Training (Stufe 2) ohne eigene Aufnahme
nachvollziehen lässt — Details dazu in `pipeline_code/README.md` → "Was ist im Repo, was
nicht".

---

## Lokal bauen & Vorschau

Voraussetzungen: [Quarto](https://quarto.org/docs/get-started/) und Python (für die
Jupyter-/Berechnungs-Zellen). Das `venv/` im Repo ist git-ignoriert; bei Bedarf neu anlegen.

```bash
# Live-Vorschau mit Auto-Reload (öffnet den Browser)
quarto preview

# Einmalig komplett rendern -> Ausgabe in _site/
quarto render

# Nur eine Seite rendern (schneller beim Iterieren)
quarto render web-ar/index.qmd
```

`freeze: auto` cached berechnete Zellen, damit unveränderte Inhalte nicht jedes Mal neu
ausgeführt werden. `_site/` ist der Build-Output und git-ignoriert — **nicht** committen.

---

## Veröffentlichen

Ein **Push auf `main`** löst `.github/workflows/publish.yml` aus: Quarto rendert die Seite und
publisht sie auf den Branch `gh-pages`, von dem GitHub Pages ausliefert. Es ist **kein**
manueller Schritt nötig.

> **Wichtig für AR:** WebXR (`immersive-ar`) braucht einen sicheren Kontext (HTTPS). Die
> GitHub-Pages-URL erfüllt das; ein lokaler `http://`-Preview kann AR nicht starten.

---

## Interaktive Bausteine (Seite „Web & AR")

Alles in `web-ar/index.qmd`. Babylon.js wird per CDN geladen (zwei `<script src=…>` ganz oben
im ersten `=html`-Block); beide Viewer teilen sich diese eine Babylon-Instanz.

1. **Orbit-Viewer** (`initSplatViewer`, Canvas `renderCanvas`) — freie Maus-/Touch-Steuerung
   um das Modell (`ArcRotateCamera`), Modellauswahl per Dropdown.
2. **AR-Demo** (`initXR`) — startet eine `immersive-ar`-Session mit Hit-Test & Platzierung.
   Läuft zuverlässig nur auf **Android-Chrome**; auf Desktop/iOS degradiert sie sauber
   (Hinweistext + QR-Code), die übrigen Viewer funktionieren überall.
3. **Begehung / „X"** (`initWalkViewer`, Canvas `walkCanvas`) — Ich-Perspektive an festen
   Standpunkten: **Ziehen** = umsehen, **Klick** = zum nächsten Standpunkt (sanft animiert).
   Funktioniert ohne AR-Hardware und ist die Demo für die Präsentation am Laptop.


## Bekannte Grenzen & Wartungs-Hinweise

- **AR nur auf Android-Chrome.** Desktop/iOS unterstützen `immersive-ar` nicht — dort
  degradiert die Demo bewusst (Hinweis + QR). Das ist erwartetes Verhalten, kein Bug.
- **Animations-`loopMode` muss `ANIMATIONLOOPMODE_CONSTANT` sein.** Beim Standpunkt-Wechsel
  (`CreateAndStartAnimation` in `goToWaypoint`) sorgt das dafür, dass die Bewegung **einmal**
  läuft und am Ziel hält. `ANIMATIONLOOPMODE_RELATIVE` (Wert `0`) würde endlos loopen und die
  Position akkumulieren — nicht zurückdrehen.
- **Eigene Tap-Erkennung statt `POINTERTAP`.** Der Klick-zum-Standpunkt nutzt bewusst
  pointerdown/up mit Bewegungs-/Zeitschwelle (statt Babylons `POINTERTAP`), weil dessen
  Doppelklick-Verzögerung Klicks verschluckt hat. So bleiben Klick (Wechsel) und Ziehen
  (Umsehen) sauber getrennt.

---

## Team & Credits

Aufgabenverteilung siehe Team-Tabelle auf `index.qmd` bzw. die Owner-Spalte in der
Projektstruktur-Tabelle oben. Repo:
[VC8-AR-Gaussian-Splatting-in-web/website](https://github.com/VC8-AR-Gaussian-Splatting-in-web/website).
