---
name: imgprompt-codex
description: Genera o modifica immagini in Codex usando i preset del programma imgprompt e lo strumento integrato image_gen. Usare per richieste come «usa il preset manga», «restaura questa foto con imgprompt», «lettering fumetto» o per elencare i preset senza chiamare le API del programma.
---

# imgprompt in Codex

Riutilizza i prompt di imgprompt tramite lo strumento integrato `image_gen`, da qualunque directory. Questa skill non esegue il wizard Python e non richiede chiavi API. Non promettere immagini illimitate o un costo specifico: limiti e fatturazione dipendono dalla modalità di accesso dell'utente.

## Scegliere e leggere il prompt

La directory delle foto e la fonte dei preset sono indipendenti. La skill è autonoma: usa [references/presets.md](references/presets.md) e, solo per le etichette interattive, [references/templates.md](references/templates.md). Risolvi questi riferimenti rispetto alla directory del presente SKILL.md, mai rispetto alla directory corrente. Non cercare `presets.py` nella cartella delle foto o in `/tmp`.

Se stai già lavorando in un checkout imgprompt identificato, oppure l'utente ne fornisce il percorso, preferisci le fonti aggiornate di quel checkout: `imgprompt/presets.py` per i cataloghi e la funzione `step_prompt` di `imgedit.py` per i template. Usa percorsi assoluti risolti dalla radice nota del checkout. Se non puoi leggerle, usa subito le copie incluse senza cercare altri checkout o richiedere accesso. Non nascondere gli errori di lettura con `2>/dev/null`: un accesso negato non equivale a un file inesistente.

I quattro cataloghi sono:

- `PRESET_PROMPTS_EDIT`: una foto, restauro, pulizia scansioni, rimozione oggetti, traduzione, manga/anime, matita, line art, outpainting e ricostruzione biglietti.
- `PRESET_PROMPTS_GENERATE`: generazione senza immagine, loghi, lettering e biglietti.
- `PRESET_PROMPTS_DUAL`: due immagini combinate.
- `PRESET_PROMPTS_MULTI`: tre o più immagini combinate.

Leggi soltanto la sezione pertinente. Non importare `imgedit.py` e non eseguire il programma per ottenere i prompt. Non leggere `.env`.

Se l'utente chiede il catalogo, elenca i preset con nomi brevi in italiano e fermati senza generare. Per una richiesta operativa, scegli il preset che corrisponde all'intento anche quando il nome è espresso in italiano. Se il nome è ambiguo e porta a risultati sostanzialmente diversi, chiedi quale risultato desidera. Se anche le copie incluse mancano o non sono leggibili, segnala il problema senza inventare il testo del preset.

Alcune voci sono etichette, non prompt completi. Per queste leggi il ramo corrispondente in [references/templates.md](references/templates.md) (o in `step_prompt` del checkout noto), cercando il nome esatto, e ricostruisci `final_prompt` con i dati dell'utente senza eseguire le domande del terminale:

- Object Removal (High Quality): oggetto da eliminare.
- Translate Text: lingua di destinazione; pulizia scansione soltanto se richiesta, altrimenti disattivata.
- A retro-style BW lettering with thick outline; 1990s Memphis Style Logo; Comic Book Style Text; Pokémon Style Lettering; Minecraft Style Lettering: testo esatto da scrivere.
- Business Card: titolo principale e dettagli esatti, mantenendo gli a capo.
- APPROVED Stamp: testo personalizzato che precede APPROVED.
- Generic Logotype: testo del logo, anche su più righe.
- Custom Prompt: istruzioni dell'utente.

Usa i dati già presenti nella conversazione; chiedi insieme solo quelli indispensabili che mancano. Non aggiungere una conferma del preset prima di generare. Mantieni i vincoli conservativi dei prompt di restauro. Le indicazioni esplicite dell'utente prevalgono sul preset: per esempio, se chiede trasparenza sostituisci il fondo verde previsto dal lettering con un fondo trasparente.

Sostituisci `{target_ratio}` con il rapporto richiesto. Per il preset che adatta il formato tramite outpainting, chiedi il rapporto se manca e non è deducibile dalla destinazione indicata. Le dimensioni in pixel o mm richieste vanno nel prompt come obiettivo: non sono parametri garantiti dello strumento. Non promettere fedeltà pixel per pixel, un vero file vettoriale, DPI certificati o un PDF pronto stampa solo sulla base dell'immagine generata.

## Generare con lo strumento integrato

Usa `image_gen` seguendo lo schema effettivamente disponibile nella sessione. Non avviare `imgedit.py`, SDK, provider OpenAI/OpenRouter/OVH o CLI API come fallback automatico. Se lo strumento non è disponibile o fallisce, comunica il limite e conserva il prompt pronto da usare; un percorso API richiede una richiesta esplicita dell'utente.

Per immagini locali, risolvi i percorsi relativi rispetto alla directory di lavoro dell'utente, verifica direttamente il file indicato (quotando i percorsi con spazi) e visualizzalo con `view_image` prima di modificarlo. Non cercare insieme foto e preset con `rg --files`: hanno fonti e percorsi distinti. Per composizioni, rendi esplicita la corrispondenza ordinata `IMG_1`, `IMG_2`, … e il ruolo di ciascuna immagine. Distingui una composizione in una sola immagine dall'applicazione separata dello stesso preset a più foto; se la richiesta non lo chiarisce, chiedilo prima di generare.

Per una nuova immagine ometti i riferimenti. Per modifiche o generazioni con riferimenti:

- Se tutti gli input hanno un percorso locale, usa `referenced_image_paths`.
- Altrimenti usa il minimo `num_last_images_to_include` che comprenda tutti gli input pertinenti, entro il limite dello strumento.
- Non passare entrambe le modalità. Se nessuna include tutti gli input richiesti, chiedi di allegare nuovamente quelli mancanti.

Imposta `transparent_background` quando richiesto e conserva la trasparenza esistente nelle modifiche, salvo indicazione contraria. Non trasformare i controlli del wizard (modello, quality, size, numero varianti) in argomenti non supportati di `image_gen`. Per più risultati richiesti, esegui chiamate distinte mantenendo il giusto insieme di riferimenti a ogni chiamata.

Per chiamate lunghe usa l'esecuzione con yield di 120 secondi prevista dallo strumento. Mostra il risultato con `generatedImage(result)` senza stampare base64. Controlla il risultato, se disponibile per ispezione, soprattutto testo, identità, composizione e vincoli richiesti; non dichiarare verifiche che non hai eseguito. Evita tentativi ripetuti senza un motivo concreto.

Se l'utente indica una destinazione, copia lì il file effettivamente restituito dallo strumento. Per elaborazioni destinate al progetto salva in `output/imagegen/` nel workspace, senza sovrascrivere originali o risultati precedenti. Se lo strumento restituisce solo un'immagine inline, mostrala e segnala che non hai un percorso locale da copiare. Per anteprime basta il risultato inline. Consegna l'immagine e, quando salvata, un link assoluto al file; indica brevemente il preset usato e rendi disponibile il prompt finale su richiesta.
