# API-Anleitung für Agenten — Reviewer Mystery-Tunes Search

Diese Datei beschreibt, **wie ein Agent (LLM, Tool, Skript) Anfragen an die laufende Such-API stellt** und welche Antworten in welchem Format zurückkommen. Stand: 2026-05-15.

---

## 1. Wohin schicken? (Endpunkte / IP / Port)

| Variante | URL | Wann verwenden |
|---|---|---|
| **Öffentlich (Cloudflare-Tunnel)** | `https://regards-financing-proxy-tony.trycloudflare.com` | Agent läuft *irgendwo im Internet* (am häufigsten der Fall). Stabile URL solange `cloudflared` läuft, kein Token nötig. |
| **Lokal** | `http://localhost:7860` | Agent läuft auf demselben Host wie der Server. |
| **Lokal über LAN/Direkt-IP** | `http://65.109.157.234:7860` | Direkt zur Maschine (nur wenn Port 7860 nach außen erreichbar wäre — derzeit gehst du normalerweise über den Cloudflare-Tunnel). |

> ⚠️ **Cloudflare-URL kann sich ändern**, wenn der `cloudflared`-Prozess neu gestartet wird. Wenn du die aktuelle URL programmatisch brauchst, lies sie aus  
> `grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /mnt/md3/spirit/laion-tunes-server/logs/cloudflared-backend.log | head -1`.

Im Folgenden steht `$BASE` für eine dieser URLs (öffentlich oder lokal).

---

## 2. Endpunkte im Überblick

| Methode | Pfad | Zweck | Body / Form |
|---|---|---|---|
| `GET`  | `/` | Liefert die Web-UI (HTML). **Nicht für Agenten.** | — |
| `GET`  | `/nsfw-report` | Interaktiver NSFW-Bericht (HTML). | — |
| `GET`  | `/api/stats` | Datensatz-Statistiken (Anzahl Tracks, Subsets, Sprachen, Index-Größen). | — |
| `POST` | `/api/search` | **Haupt-Suche**: Vector / BM25 / Combined über Text-Query, mit Filtern und optionaler 2-Phasen-Verfeinerung. | JSON |
| `POST` | `/api/search_similar` | "Ähnliche Tracks zu Track `row_id` finden" (über vorberechnete Whisper-Audio-Embeddings). | JSON |
| `POST` | `/api/search_by_audio` | Eigene Audio-Datei hochladen, ähnliche Tracks finden (über Whisper-Encoder). | `multipart/form-data` |

> ℹ️ **Aktuell geladene Indizes** (Memory-sparsame Konfiguration, Stand 2026-05-15):  
> ✅ `caption` (1,356,009 Vektoren), `transcription` (1,041,488), `whisper` (402,649), BM25 `caption` (1,356,009), SQLite (1,429,734 Tracks), Instrumental-Subset (11,391 Tracks).  
> ⏭ Aktuell **nicht** geladen: FAISS `tag` / `lyric` / `mood`, BM25 `tags` / `transcription`.  
> Anfragen, die einen nicht geladenen Index brauchen, geben leere Ergebnisse zurück (kein Fehler).  
> Beste Suchqualität: `search_type=vector`, `vector_field=caption`.

---

## 3. `GET /api/stats`

Keine Parameter. Antwort als JSON:

```json
{
  "total_tracks": 1429734,
  "subsets": { "mureka": 383549, "suno": 1037381, "udio": 8804 },
  "score_average": { "mean": 3.29, "min": 1.40, "max": 4.77 },
  "with_caption": 1356009,
  "with_transcription": 1041488,
  "faiss_indices": { "caption": 1356009, "transcription": 1041488, "whisper": 402649 },
  "bm25_indices":  { "caption": 1356009 },
  "languages":     { "en": 700000, "unknown": 350000, "es": 90000, ... },
  "instrumental_count": 388301,
  "whisper_embeddings": 402649,
  "instrumental_subset_tracks": 11391
}
```

**Verwendung im Agent**: zur Validierung, dass der Server lebt und welche Indizes verfügbar sind, bevor du `search_type` / `vector_field` wählst.

Beispiel (Python):

```python
import requests
stats = requests.get(f"{BASE}/api/stats", timeout=60).json()
assert stats["total_tracks"] > 0
available_vectors = list(stats["faiss_indices"].keys())   # z.B. ["caption","transcription","whisper"]
```

> Erstaufruf ist langsam (~30 – 60 s; kalte SQLite-Caches). Folgeaufrufe sind schnell.

---

## 4. `POST /api/search` — Haupt-Suche

**Content-Type:** `application/json`. Body = JSON-Objekt mit den folgenden Feldern. Alle Felder außer `query` haben Defaults.

### 4.1 Request-Felder

| Feld | Typ | Default | Bedeutung |
|---|---|---|---|
| `query` | `string` | **Pflicht** | Suchtext in natürlicher Sprache. |
| `negative_query` | `string \| null` | `null` | Negativ-Prompt (wird im Vector-Modus vom Query-Vektor subtrahiert). |
| `search_type` | `string` | `"bm25"` | `"vector"`, `"bm25"`, oder `"combined"`. **Empfehlung: `"vector"`.** |
| `vector_field` | `string` | `"caption"` | FAISS-Index für Vector-Suche: `"caption"`, `"transcription"`, `"tag"`, `"lyric"`, `"mood"`. (Aktuell nur `caption`/`transcription` voll abgedeckt; `tag`/`lyric`/`mood` derzeit nicht geladen.) |
| `bm25_field` | `string` | `"caption"` | BM25-Index: `"caption"`, `"tags"`, `"transcription"`, `"lyrics_hashed"`. (Aktuell nur `caption` voll abgedeckt.) |
| `rank_by` | `string` | `"similarity"` | `"similarity"`, `"aesthetics"`, `"plays"`, `"likes"`. |
| `min_aesthetics` | `float \| null` | `null` | Filter: Mindest-Score (0 – 5). |
| `min_similarity` | `float \| null` | `null` | Filter: Mindest-Cosinus-Ähnlichkeit. |
| `subset_filter` | `string \| null` | `null` | `"suno"`, `"udio"`, `"mureka"`, `"no_riffusion"`, `"instrumental_subset"` etc. |
| `vocal_filter` | `string \| null` | `null` | `"instrumental"` oder `"vocals"`. |
| `min_duration` | `float \| null` | `60.0` | Sekunden. Filter `≥` Dauer. Auf `0` setzen, wenn auch sehr kurze Tracks erlaubt sind. |
| `languages` | `list[str] \| null` | `null` | Liste von Sprachcodes (z. B. `["en","es"]`). `null` = alle. |
| `negative_weight` | `float` | `0.7` | Gewicht der Negativ-Subtraktion (0 – 1). |
| `nsfw_filter` | `string \| null` | `null` | `"sfw_only"`, `"nsfw_only"` oder `null`. |
| `top_k` | `int` | `50` | Anzahl Ergebnisse. |
| `stage2_enabled` | `bool` | `false` | Zwei-Phasen-Suche aktivieren. |
| `stage2_query` | `string \| null` | `null` | Query für Phase 2 (Re-Ranking). |
| `stage2_search_type` | `string` | `"vector"` | `"vector"` oder `"bm25"`. |
| `stage2_vector_field` | `string` | `"caption"` | Wie `vector_field`. |
| `stage2_bm25_field` | `string` | `"caption"` | Wie `bm25_field`. |
| `stage2_top_k` | `int` | `50` | Anzahl Ergebnisse nach Phase 2. |

### 4.2 Beispiel-Request (curl)

```bash
curl -X POST "$BASE/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "dreamy ambient synth pad",
    "search_type": "vector",
    "vector_field": "caption",
    "rank_by": "similarity",
    "nsfw_filter": "sfw_only",
    "min_duration": 60,
    "top_k": 10
  }'
```

### 4.3 Beispiel-Request (Python)

```python
import requests

resp = requests.post(
    f"{BASE}/api/search",
    json={
        "query": "dreamy ambient synth pad",
        "search_type": "vector",
        "vector_field": "caption",
        "rank_by": "similarity",
        "nsfw_filter": "sfw_only",
        "min_duration": 60,
        "top_k": 10,
    },
    timeout=120,   # erste Anfrage kann 30 - 60 s dauern (kalter Cache)
)
resp.raise_for_status()
data = resp.json()
for hit in data["results"]:
    print(hit["row_id"], hit["score"], hit["title"], hit["audio_url"])
```

### 4.4 Response-Format

```json
{
  "results": [ /* Liste von Track-Objekten, siehe Abschnitt 7 */ ],
  "total_candidates": 2000,         // Anzahl Kandidaten vor Filtern
  "total_filtered": 1847,           // Anzahl nach Filtern
  "total_tracks": 1429734,          // Gesamt-Dataset
  "search_time_ms": 245.3,
  "query_embedding_time_ms": 28.5,
  "search_type": "vector",
  "vector_field": "caption",
  "bm25_field": "caption",
  "stage2": { /* optional, nur wenn stage2_enabled = true */ }
}
```

---

## 5. `POST /api/search_similar` — Ähnliche Tracks zu einer `row_id`

Findet Tracks, die einem schon im Datensatz vorhandenen Track audio-ähnlich klingen (FAISS-Lookup über die vorberechneten Whisper-Audio-Embeddings).

### 5.1 Request

```json
{
  "row_id": 627211,
  "top_k": 20,
  "rank_by": "similarity",
  "min_aesthetics": null,
  "subset_filter": null,
  "vocal_filter": null,
  "min_duration": 60.0,
  "languages": null,
  "nsfw_filter": "sfw_only",
  "stage2_enabled": false,
  "stage2_query": null,
  "stage2_search_type": "vector",
  "stage2_vector_field": "caption",
  "stage2_bm25_field": "caption",
  "stage2_top_k": 50
}
```

`row_id` ist Pflicht und muss eine ID sein, für die ein Whisper-Embedding existiert (`has_whisper_emb=true` im Track-Objekt; aktuell 402,649 Tracks).

### 5.2 Response

Gleiche Struktur wie `/api/search`, zusätzlich:
- `search_type: "music_similarity"`
- `vector_field: "whisper"`
- `reference_row_id`: die übergebene `row_id`
- `reference_title`: Titel des Referenz-Tracks

### Fehler

- `503` — Whisper-Index nicht geladen
- `404` — `row_id` hat kein Whisper-Embedding

---

## 6. `POST /api/search_by_audio` — Audio hochladen

Lädt eine Audio-Datei hoch, lässt sie durch den Music-Whisper-Encoder laufen, und findet damit ähnliche Tracks.

> ⚠️ **Aktueller Status:** Der Server läuft mit `--no-whisper`, weil der Encoder zusätzlich ~500 MB RAM braucht und wir derzeit memory-constrained sind. Anfragen an diesen Endpunkt liefern aktuell `503 Whisper encoder not loaded`. Endpunkt wird hier dokumentiert, weil die Spec-Stabilität wichtig ist.

### Request: `multipart/form-data`

| Feld | Typ | Pflicht | Bemerkung |
|---|---|---|---|
| `audio` | Datei | ja | mp3 / wav / flac / ogg / m4a etc. Max 100 MB. Erste 30 Sek. werden verwendet. |
| `top_k` | int | nein | Default 50 |
| `rank_by` | string | nein | wie `/api/search` |
| `subset_filter` | string | nein | |
| `vocal_filter` | string | nein | |
| `min_duration` | float | nein | |
| `min_aesthetics` | float | nein | |
| `languages` | string | nein | Komma-separiert, z. B. `"en,es"` |
| `nsfw_filter` | string | nein | |
| `stage2_enabled` | string | nein | `"true"` zum Aktivieren |
| `stage2_query` | string | nein | |
| `stage2_search_type` | string | nein | |
| `stage2_vector_field` | string | nein | |
| `stage2_bm25_field` | string | nein | |
| `stage2_top_k` | int | nein | |

### Beispiel (curl)

```bash
curl -X POST "$BASE/api/search_by_audio" \
  -F "audio=@my_song.mp3" \
  -F "top_k=20" \
  -F "nsfw_filter=sfw_only"
```

### Beispiel (Python)

```python
with open("my_song.mp3", "rb") as f:
    resp = requests.post(
        f"{BASE}/api/search_by_audio",
        files={"audio": ("my_song.mp3", f, "audio/mpeg")},
        data={"top_k": 20, "nsfw_filter": "sfw_only"},
        timeout=180,
    )
```

### Response

Gleiche Struktur wie `/api/search`, zusätzlich:
- `search_type: "music_similarity"`
- `vector_field: "whisper"`
- `audio_filename`: ursprünglicher Dateiname
- `cache_hit`: `bool` (Embedding-Cache pro Client-IP+Filename, 1 Stunde TTL)

### Fehler

- `503` — Whisper-Encoder nicht geladen (aktueller Default)
- `400` — leere Audio-Datei, > 100 MB, oder Audio-Verarbeitung gescheitert

---

## 7. Track-Objekt (Result-Schema)

Jeder Eintrag in `data["results"]` hat diese Form:

```json
{
  "row_id": 627211,
  "title": "Dramatic Orchestral Grandeur",
  "audio_url": "https://cdn1.suno.ai/abc123.mp3",
  "subset": "suno",                              // "suno" | "udio" | "mureka"
  "tags_text": "orchestral, cinematic, epic",
  "mood_text": "dramatic, sweeping",
  "genre_tags": ["Classical","Soundtrack / Score"],
  "scene_tags": [],
  "emotion_tags": ["Awe"],
  "score_average": 3.84,
  "score_coherence": 3.9, "score_musicality": 3.7,
  "score_memorability": 3.8, "score_clarity": 3.9,
  "score_naturalness": 3.8,
  "play_count": 1240, "upvote_count": 89,
  "duration_seconds": 178.6,
  "music_whisper_caption": "The listener hears a piece of music ...",
  "has_caption": true,
  "has_transcription": false,
  "is_instrumental": true,
  "language": "unknown",
  "score": 0.513,
  "score_type": "cosine_similarity",            // cosine_similarity | bm25 | aesthetics | play_count | upvote_count
  "has_whisper_emb": true,                      // dann auch /api/search_similar möglich
  "nsfw_overall_label": "likely_sfw",
  "nsfw_gore_label": "likely_sfw",
  "nsfw_sexual_label": "likely_sfw",
  "nsfw_hate_label": "likely_sfw",
  "nsfw_gore_sim": 0.21,
  "nsfw_sexual_sim": 0.19,
  "nsfw_hate_sim": 0.20
}
```

Bei aktiver Zwei-Phasen-Suche zusätzlich:
- `stage1_score`: Score aus Phase 1
- `stage2_score`: Score aus Phase 2

> Die Audio-Datei selbst wird *nicht* vom Server ausgeliefert — `audio_url` zeigt auf das CDN des Original-Anbieters (Suno / Udio / Mureka). Der Agent lädt von dort.

---

## 8. Praktische Hinweise für Agenten

1. **Timeouts.** Erstaufrufe nach Server-Neustart sind langsam (kalte SQLite/FAISS-Caches), oft 30 – 60 s. Setze in deinem HTTP-Client mindestens `timeout=120` für die ersten paar Calls und `timeout=30` für warme.

2. **Empfohlener Default für freie Text-Anfragen**:  
   ```json
   {
     "query": "<deine Anfrage>",
     "search_type": "vector",
     "vector_field": "caption",
     "rank_by": "similarity",
     "nsfw_filter": "sfw_only",
     "min_duration": 60,
     "top_k": 20
   }
   ```
   Das nutzt den am vollständigsten abgedeckten FAISS-Index (`caption`, 1.36 M Tracks) mit dem korrekt geladenen PyTorch-EmbeddingGemma.

3. **Health-Check vor Bulk-Anfragen**:  
   ```python
   try:
       stats = requests.get(f"{BASE}/api/stats", timeout=120).json()
       ok = stats["total_tracks"] > 1_000_000 and "caption" in stats["faiss_indices"]
   except Exception:
       ok = False
   ```

4. **Fehlerformate.** Bei `400 / 404 / 503` ist der Body normalerweise:  
   ```json
   { "error": "Whisper encoder not loaded" }
   ```

5. **Rate-Limit.** Es gibt keinen App-seitigen Rate-Limit, aber der Server hat **einen einzigen Uvicorn-Worker**: parallele Calls werden sequentialisiert. Plane für Bulk-Anfragen mit `min(parallel) = 1`, sonst stauen sich Requests.

6. **Stabilität.** Die Box hat keinen Swap; der FastAPI-Prozess wurde in den letzten Tagen einige Male nachts OOM-gekillt und neu gestartet. Wenn du eine `502` / Timeout vom Tunnel siehst, prüfe noch einmal nach 1 – 2 Min. — wahrscheinlich ist gerade ein Restart im Gange.

7. **Audio-Wiedergabe / Download.** Die `audio_url` zeigt auf öffentlich abrufbare CDN-Endpunkte. Suno-URLs (`cdn1.suno.ai/...mp3`) sind direkt streambar; Mureka-URLs (`static-cos.mureka.ai/...mp3`) ebenso; Udio-URLs (`storage.googleapis.com/udio-artifacts-...`) ebenso.

---

## 9. Speicherort dieser Datei

```
/mnt/md3/spirit/laion-tunes-server/API_AGENT_GUIDE.md
```

Diese Datei wird **nicht automatisch** in den HF-Repo hochgeladen — sie ist lokal. Wenn ein Agent sie braucht, kann er sie z. B. über:

```bash
cat /mnt/md3/spirit/laion-tunes-server/API_AGENT_GUIDE.md
```

einlesen, oder du gibst sie ihm direkt als Kontext mit.
