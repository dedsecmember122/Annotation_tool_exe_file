# Changelog

Running log of fixes and feature work on the annotation tool, kept by date.

> **A note on completeness:** this log started on 2026-07-19. Everything
> dated before that is a **baseline reconstructed by reading the current
> code** (comments, file layout, mtimes) — there's no chat history or
> per-commit git log to draw on for that period (this repo has a single,
> unrelated commit; earlier assistant sessions weren't recorded to memory).
> That means:
> - Feature entries in the baseline are reliable — a feature either exists
>   in the code or it doesn't.
> - Fix entries in the baseline are a **lower bound** — only fixes the code
>   happened to leave a comment about are listed. Undocumented fixes from
>   that period leave no trace and aren't counted.
> - From 2026-07-19 onward, entries are logged as the work happens, so
>   those counts are exact.

---

## Totals

| Period | Feature additions | Bug fixes |
|---|---|---|
| Baseline (2026-07-12 → 2026-07-18, reconstructed) | 9 major subsystems | 5 documented in code |
| 2026-07-19 (logged live) | 1 | 3 |
| 2026-07-20 (logged live) | 2 | 3 |
| 2026-07-21 (logged live) | 5 | 8 |
| 2026-07-22 (logged live) | 1 | 0 |
| 2026-07-23 (logged live) | 0 | 1 |

---

## 2026-07-23

**0 feature additions, 1 bug fix**

### Bug fixes

1. **Training failed on every installed .exe with `[WinError 267] The
   directory name is invalid` (root cause)** — `backend/app/core/config.py`,
   `build/annotation_tool.spec`, `backend/app/ml/custom_model_adapter.py`,
   `frontend/main.py`, `detc-core/train.py`
   Reported by a real customer after installing the app. `CUSTOM_MODEL_DIR`
   (the folder holding the DETC training/inference code) defaulted to
   `Path(__file__).resolve().parent.parent.parent.parent / "detc-core"` —
   in dev mode that lands next to `backend/` correctly, but in a frozen
   PyInstaller build `__file__` resolves under the temporary extraction
   folder (`sys._MEIPASS`) instead, and `detc-core/` was never added to
   `annotation_tool.spec`'s `datas`, so that folder never actually existed
   in any packaged build. Training passed this nonexistent path straight to
   `subprocess.Popen(cwd=...)`, which Windows rejects with exactly
   `[WinError 267]` — reproduced locally to confirm the exact mechanism.
   This also meant model *inference* (auto-annotate), not just training,
   has never worked in a shipped exe, since `load()` uses the same setting.
   A second, latent bug would have surfaced immediately after fixing the
   first: `sys.executable` in a frozen build is the app's own exe (there's
   no bundled `python.exe` to hand `train.py` to), so the existing
   `[sys.executable, train_script, ...]` command would have just launched a
   second copy of the GUI instead of running training.
   Fixed by: (1) bundling `detc-core/` into the PyInstaller `datas` so it's
   actually present in the packaged build; (2) making `CUSTOM_MODEL_DIR`
   anchor on `sys._MEIPASS` when frozen, the same fix already applied to
   resource lookups in `frontend/main.py`'s `_resource_path()`; (3) adding a
   hidden `--train-worker` mode to `frontend/main.py` so a frozen build can
   re-invoke itself as the training subprocess and dispatch straight into
   `train.py`'s training loop instead of relying on a nonexistent bundled
   Python interpreter (the same trick `multiprocessing.freeze_support()`
   uses); (4) a pre-flight check before `subprocess.Popen` that raises a
   clear, actionable message ("Training code directory not found... try
   reinstalling") if the model directory is ever missing again, instead of
   a bare Windows error code; (5) attaching the last ~20 lines of raw
   subprocess output to the "no checkpoint found" error so a training
   failure from any other cause is diagnosable from the UI log instead of
   just a return code.

---

## 2026-07-22

**1 feature addition, 0 bug fixes**

### Feature additions

1. **macOS build (.app + .dmg)** — `build/annotation_tool.spec`,
   `.github/workflows/build-macos.yml` (new), `frontend/resources/icon.icns`
   (new)
   The app previously only shipped as a Windows exe/installer, despite
   being built on cross-platform tech (Qt, FastAPI, SQLite) that already
   ran fine in dev mode on macOS. `annotation_tool.spec` is now shared by
   both platforms — it detects macOS at build time and wraps the same
   PyInstaller output in a proper `.app` bundle (icon, `Info.plist`,
   bundle identifier) instead of a bare Windows `.exe`, using a matching
   `.icns` generated from the same source artwork as the Windows `.ico`.
   A new `build-macos.yml` CI workflow (mirroring `build-windows.yml`)
   builds it on a `macos-latest` runner and packages a `.dmg` via
   `create-dmg` for a drag-to-Applications install, matching the Windows
   installer's "no CLI" bar.
   Unlike the Windows build (which I could only validate through CI, two
   rounds of trial-and-error), this one was fully tested locally first —
   this machine is a Mac. Ran a real `pyinstaller build/annotation_tool.spec`
   locally, inspected the resulting `.app`'s `Info.plist`, actually
   launched it, and confirmed the embedded backend came up and answered
   `/health` in the real frozen build (not dev mode) before touching CI.
   Note: GitHub's `macos-latest` runners are Apple Silicon (arm64) — this
   build only runs natively on Apple Silicon Macs, not Intel ones; a
   universal2 build isn't practical since not every dependency (torch
   included) ships a universal2 wheel.

---

## 2026-07-21

**5 feature additions, 8 bug fixes**

Vedant's first hands-on test pass surfaced the last three items below: admins
couldn't provision teammates, Export had no dataset-split option, and the
image gallery visually looked broken.

The first two bug fixes below were found from the first real test of the
packaged `.exe` on a machine other than the dev machine — the embedded
backend never came up, showing "The embedded backend failed to start
within 15 seconds. Check that port 8765 is not in use."

### Feature additions

1. **Admin Dashboard + role management** — `frontend/ui/admin_dashboard.py`
   (new), `frontend/ui/main_window.py`, `frontend/api_client.py`,
   `backend/app/api/auth.py`, `backend/app/schemas/schemas.py`
   Previously the only way to become admin was being the very first person
   to ever sign up, and the admin API endpoints (list users, extend/reset
   trial) existed but had no UI — using them meant calling the API
   directly. Added a new `POST /admin/users/{id}/role` endpoint to
   promote/demote any user between `admin`/`annotator` (an admin cannot
   change their own role — that has to be done by a different admin, so an
   account can never demote itself into leaving zero admins), and a new
   **Admin Dashboard** window (File → Admin Dashboard, admin-only) listing
   every user with their role and trial status, with buttons to
   promote/demote, extend trial +7 days, or reset trial — wired to the
   admin endpoints that already existed in `api_client.py` but nothing
   previously called.

2. **Windows installer wizard** — `build/installer.iss` (new),
   `.github/workflows/build-windows.yml`
   Previously the only distributable was the raw `AnnotationTool.exe` —
   fine for a quick test, but no Start Menu entry, no desktop shortcut, and
   no clean way to uninstall short of deleting the file. Added an Inno
   Setup script that wraps the PyInstaller output into a standard installer
   wizard (license → install location → shortcuts → install), installing
   per-user under `%LocalAppData%\Programs\AnnotationTool` so it needs no
   admin rights. The CI build now produces both artifacts —
   `AnnotationTool-windows` (raw exe) and `AnnotationTool-windows-installer`
   (`AnnotationToolSetup.exe`) — and tag-triggered releases attach both.

3. **Rebrand to "InSiSo Model Bench" + real app icon** —
   `frontend/main.py`, `frontend/ui/main_window.py`,
   `frontend/ui/login_window.py`, `frontend/ui/signup_window.py`,
   `frontend/resources/icon.ico` (new), `build/annotation_tool.spec`,
   `build/installer.iss`
   The app previously showed as "AnnotationTool" everywhere (window
   titles, taskbar, splash screen) with no custom icon — just Qt/PyInstaller
   defaults. Added a proper `.ico` (generated from the provided artwork, 7
   embedded resolutions from 16×16 to 256×256) and wired it into the exe
   itself (`annotation_tool.spec`'s `icon=`), the app-wide `QApplication`
   window icon (so every window — login, main, dialogs — picks it up
   without setting it individually), and the installer's own icon
   (`installer.iss`'s `SetupIconFile`). Renamed the app-facing name to
   "InSiSo Model Bench" in the taskbar/window titles, splash screen, and
   installer (Start Menu group, shortcuts, install folder name) — the
   underlying `.exe` filename and internal `%USERPROFILE%\AnnotationTool\`
   data folder were deliberately left as-is, to avoid orphaning any
   existing installs' local database/storage. Also set the installer's
   desktop-icon checkbox to checked by default, since a discoverable
   desktop icon was the point.

4. **Admins can create user accounts directly** — `backend/app/api/auth.py`
   (new `POST /admin/users`), `backend/app/schemas/schemas.py`,
   `frontend/api_client.py`, `frontend/ui/admin_dashboard.py`
   The only way to create an account was public self-signup — an admin had
   no way to provision a teammate's login directly. Added an admin-only
   create-user endpoint (username/email/password/role, skips the
   confirm-password step self-signup needs) and a "+ Add User" button in
   the Admin Dashboard opening a small form for it.

5. **Export: split into Train / Validation / Test sets** —
   `backend/app/api/export.py`, `frontend/api_client.py`,
   `frontend/ui/export_dialog.py`
   Export previously always produced one flat file/folder — no way to
   carve it into the train/val/test partitions a model-training workflow
   actually needs. Added an optional split (checkbox + Train%/Val% spinners
   in the Export dialog, Test% is the remainder) built on the same
   shuffle-and-cutoff approach `CustomModelAdapter` already uses internally
   for its own train/val split, extended to a third bucket. Applies to
   every export format: COCO becomes `train.json`/`val.json`/`test.json` in
   a zip, YOLO and Pascal VOC get `train/`/`val/`/`test/` subfolders (with
   `classes.txt`/shared class list once at the root, not duplicated per
   split), and the full dataset zip splits both
   `images/{split}/` and `annotations/{split}.json`. Unsplit exports (the
   default) are completely unchanged — verified directly against the API
   with a real 10-image project, including that an invalid split
   (percentages summing over 100%) is rejected with a clear 400 instead of
   silently producing something wrong.

### Bug fixes

1. **Embedded backend crash in the packaged .exe (root cause)** —
   `frontend/main.py`
   A windowed (`console=False`) PyInstaller build has no console attached,
   so `sys.stdout`/`sys.stderr` are `None`. Uvicorn's logging setup calls
   `sys.stdout.isatty()` to decide whether to colorize output, which
   crashed with `AttributeError: 'NoneType' object has no attribute
   'isatty'` on every single launch of the packaged exe — this was the
   actual cause of the backend never starting, not a timing or port issue.
   Fixed by patching `sys.stdout`/`sys.stderr` to `os.devnull` streams at
   the very top of the entry point, before uvicorn (or anything else) can
   touch them.

2. **Backend startup failures were invisible, and timed out slowly** —
   `frontend/main.py`
   Two compounding issues found while diagnosing bug fix #1: (a) the
   background backend thread had no error handling, so any exception in it
   just vanished — the user only ever saw a generic "check port 8765"
   guess with no real diagnostic. (b) the health-check wait always ran the
   full timeout even after the backend thread had already died, and the
   fixed 15s timeout didn't leave room for onefile PyInstaller builds'
   slower cold start (re-extracting the whole bundle, including torch, on
   every launch) on an unfamiliar machine. Fixed by: logging any real
   exception to `%USERPROFILE%\AnnotationTool\backend_error.log` and
   surfacing that path in the error dialog; returning as soon as the
   backend thread dies instead of waiting out the rest of the timeout; and
   raising the timeout to 90s for frozen (packaged) builds specifically.
   This is what surfaced the actual traceback for bug fix #1 above.

3. **Trial expiry was only ever checked once, at first login** —
   `backend/app/api/auth.py`
   Found while explaining the trial-enforcement logic. `login()` correctly
   blocks an expired trial, but access tokens are short-lived by design
   (60 min) and refresh tokens last 7 days — and `refresh()` never
   re-checked expiry, so a user already logged in when their trial expired
   could keep silently refreshing straight through it for up to 7 more
   days. Fixed by applying the same expiry check in `refresh()`.

4. **Login/signup windows had no way to minimize, and login had no way to
   close, the app** — `frontend/ui/window_controls.py` (new),
   `frontend/ui/login_window.py`, `frontend/ui/signup_window.py`
   Both windows use `Qt.FramelessWindowHint` for their custom rounded-card
   look, which drops the native title bar — and with it, every native
   button (minimize, maximize, close) — entirely, leaving no visible way to
   minimize the app or (on the login screen specifically) close it. Added a
   small shared `make_window_controls()` helper providing custom minimize
   (−) and close (✕) buttons styled to match the card, added to both
   windows' custom header. Maximize was intentionally left out — both are
   fixed-size forms where maximizing wouldn't change anything.

5. **Image gallery: photos overlapping/merging together** —
   `frontend/ui/project_dashboard.py`
   The gallery declares a fixed 160×120 icon box (`setIconSize`), but
   server thumbnails are resized to a max of 240px on their *longest* side
   (see the 2026-07-20 thumbnail-endpoint entry) — so depending on an
   image's aspect ratio, the actual thumbnail could be noticeably bigger
   than the box it was placed in (e.g. 240×200), and since it was never
   scaled down to fit before being set as the item's icon, it overflowed
   past its grid cell into neighboring photos. Reproduced offscreen with a
   set of mismatched-aspect-ratio thumbnails to confirm before fixing —
   fixed by scaling each thumbnail to the gallery's icon size
   (aspect-preserving) before setting it, confirmed clean afterward.

6. **CI build broke after adding the app icon** — `build/annotation_tool.spec`
   PyInstaller resolves a bare relative `icon=` path against the `.spec`
   file's own directory (`build/`), not the directory `pyinstaller` was
   invoked from — unlike `datas`, which already used an absolute
   `ROOT`-based path for exactly this reason. `icon="frontend/resources/icon.ico"`
   was looked up as `build/frontend/resources/icon.ico`, which doesn't
   exist, failing the build with `FileNotFoundError`. Fixed by using the
   same `ROOT`-based absolute path as `datas`.

7. **Theming (and the window icon) silently failed to load in the packaged
   exe** — `frontend/main.py`
   Found while investigating Vedant's report of a near-unreadable, oddly
   light-and-dark-mixed login screen (white input boxes and an unstyled
   button against an otherwise dark card). Root cause: `_apply_theme()`
   located `style_dark.qss` via `Path(__file__).parent / "resources"` —
   but `main.py` is the PyInstaller *entry script*, and PyInstaller has a
   real footgun where an entry script's `__file__` resolves to a flattened
   path inside the frozen bundle that's missing the `frontend/` prefix the
   bundled data actually lives under (unlike `__file__` for normally
   *imported* modules, which keeps the full package path). The lookup
   silently failed — no exception, `_apply_theme()` just skips applying
   anything when the file isn't found — so only widgets with inline
   Python-set styles (the login card's own background, a few labels)
   rendered correctly, while everything depending on the external
   stylesheet (input fields, the primary button, etc.) fell back to
   native/unstyled rendering. The runtime window icon lookup had the same
   bug. Fixed both via a new `_resource_path()` helper that anchors on
   `sys._MEIPASS` (PyInstaller's guaranteed extraction root) when frozen,
   instead of a script's own `__file__`. Verified against a simulated
   frozen `_MEIPASS` layout, since this can't be exercised in dev mode
   (where the bug doesn't reproduce).

8. **Login/signup windows: bottom row clipped/overlapping after adding the
   minimize/close controls** — `frontend/ui/login_window.py`,
   `frontend/ui/signup_window.py`
   Adding the custom title-bar row (see the window-controls fix above)
   added height to the card's content without increasing the dialog's own
   `setFixedSize()` — in a fixed-size window, content taller than
   available space gets compressed, which pushed the "Create account" /
   "Sign in" row into the row above it. Reproduced offscreen (rendered the
   actual dialog and inspected the output) to confirm before fixing;
   fixed by increasing both dialogs' fixed height by 50px and re-verified
   clean.

> **Testing note:** the free-trial period was temporarily shortened from
> 7 days to 1 (`TRIAL_PERIOD_DAYS` in `backend/app/models/models.py`) so
> expiry behavior could actually be tested without waiting a week. This is
> explicitly marked in code as a testing value — revert to 7 (or whatever
> the real policy ends up being) before rolling out to real customers. Not
> counted in the totals above, since it isn't a feature or a bug fix.

---

## 2026-07-20

**2 feature additions, 3 bug fixes**

### Feature additions

1. **Mark auto-annotations as reviewed** — `frontend/tools/bbox_tool.py`,
   `frontend/tools/polygon_tool.py`, `frontend/ui/annotation_canvas.py`,
   `frontend/ui/main_window.py`
   Closes the gap noted in the 2026-07-19 baseline: the frontend now
   actually reads and writes `Annotation.reviewed`. Correcting an
   auto-annotated box or polygon (move/resize) now automatically flips it
   to reviewed, since fixing it *is* the review. For auto-annotations that
   are already correct, a new "✓ Mark Reviewed" button in the layers panel
   marks them reviewed without requiring an edit. The layers list now shows
   each annotation's review state (`auto ✓` / `auto ⚠ needs review`), and
   the orange "needs review" box color now reflects the real per-item
   state instead of always showing for every auto box regardless of
   whether it had been reviewed.

2. **Server-resized image thumbnails** — `backend/app/api/images.py` (new
   `GET /images/{id}/thumbnail`), `frontend/api_client.py`,
   `frontend/ui/project_dashboard.py`
   New backend endpoint resizes images server-side (PIL, max 240px, JPEG)
   instead of the client downloading and decoding full-resolution
   originals just to shrink them to a 160x120 gallery icon. Cuts a typical
   photo from ~megabytes down to ~1-2KB transferred per thumbnail. Gallery
   thumbnail loading also moved off the UI thread onto a background worker,
   so the list of images now appears immediately and icons fill in
   progressively instead of freezing the window. See bug fix #1 below for
   why this mattered.

### Bug fixes

1. **App became slow/heavy shortly after opening a project (M1 report)** —
   `frontend/ui/project_dashboard.py`
   The project gallery fetched the full-resolution original of *every*
   image in the project, synchronously on the UI thread, just to build a
   160x120 thumbnail — and this reran on every project open, status-filter
   change, image upload/delete, and every return from the annotation view
   (`main_window.py`'s `refresh()` fires on every "back to dashboard").
   For a real project (many, often multi-megapixel images) this both froze
   the UI and spiked memory decoding full images repeatedly, which reads
   exactly like the reported "slow after opening" / memory issue. Fixed by
   the new thumbnail endpoint + async loading above.

2. **Manually-drawn boxes could have `reviewed` silently reset to false** —
   `frontend/ui/annotation_canvas.py`
   The backend auto-marks manually-created annotations `reviewed=True` at
   creation, but the frontend never synced that back onto the local item
   after `create_annotation()` — so the item kept its local default of
   `False`, and the *next* save (e.g. the 30s autosave) would push that
   stale `False` back over the server's `True`. Found while wiring up the
   reviewed flag on save; fixed by syncing `item.reviewed` from the
   create response.

3. **Moving a polygon never saved** — `frontend/tools/polygon_tool.py`
   `PolygonItem` had no `mousePressEvent`/`mouseReleaseEvent` overrides, so
   dragging a polygon as a whole never marked the canvas dirty — the same
   bug class as the bbox "edited boxes were silently not saved" fix from
   2026-07-19, just not previously noticed because polygons have no resize
   handles, making whole-polygon moves the only (and rarer) edit path.
   Fixed alongside the reviewed-flag work by mirroring `BBoxItem`'s
   move-tracking.

---

## 2026-07-19

**1 feature addition, 3 bug fixes**

### Feature additions

1. **Resizable bounding boxes** — `frontend/tools/bbox_tool.py`
   Selected boxes now show 8 draggable resize handles (4 corners + 4 edge
   midpoints), with per-handle hover cursors. Dragging a handle resizes the
   box and is undoable (Ctrl+Z), reusing the `ResizeBBoxCommand` that
   existed in `undo_redo_manager.py` but was never wired up. Previously a
   box could only be moved as a whole — there was no way to adjust its
   size, which was the main gap in editing auto-annotated boxes.

### Bug fixes

1. **Edited boxes were silently not saved** — `frontend/tools/bbox_tool.py`
   Moving or resizing an existing box never marked the canvas as "dirty",
   so neither the 30s autosave nor Ctrl+S would persist the change — edits
   were lost on navigating away. Both actions now correctly flag the
   canvas for saving.

2. **Dark/Light theme toggle button did nothing** — `frontend/main.py`
   The app runs as `python frontend/main.py`, i.e. as `__main__`. The
   toggle button's handler imported `frontend.main` as a *second*, separate
   module copy whose own module-level `_app` reference was never
   initialized, so clicking the button called `.setStyleSheet()` on `None`
   and failed silently. Fixed `_apply_theme()` to look up the live
   `QApplication.instance()` instead of the module-level global, so it
   works regardless of which module copy calls it.

3. **App crash (segfault) when navigating away with a box selected** —
   `frontend/ui/annotation_canvas.py`
   Switching images, or going back to the dashboard, while an annotation
   was still selected could crash the whole app. `scene().clear()` destroys
   every item in C++ immediately; if a selected item was among those being
   destroyed, Qt re-entered Python via a reentrant `selectionChanged` signal
   while that item was still mid-destruction, and resolving the dangling
   reference segfaulted the process. Reproduced directly (confirmed SIGSEGV)
   and fixed by clearing the scene's selection *before* destroying items, so
   the signal fires while everything is still valid.
   This was a pre-existing bug, surfaced more easily now that selecting a
   box (to use the new resize handles) is much more common.

---

## Baseline (2026-07-12 → 2026-07-18) — reconstructed from code

### Feature additions

1. **Accounts & auth** — `backend/app/api/auth.py`, `login_window.py`, `signup_window.py`
   Signup/login with JWT access + refresh tokens, role-based access
   (`admin`/`annotator`), and a 7-day free trial system for non-admin users
   (auto-stamped on first login) with admin-only endpoints to extend or
   reset a user's trial.

2. **Project workspace** — `frontend/ui/project_dashboard.py`, `backend/app/api/projects.py`
   Multi-project management, threaded image upload (non-blocking UI),
   status-filtered image gallery (unannotated / in progress / annotated /
   auto-annotated), bulk image deletion.

3. **Class manager** — `frontend/ui/class_manager_dialog.py`
   Per-project label classes with assignable colors.

4. **Annotation canvas** — `frontend/ui/annotation_canvas.py`, `frontend/tools/*`
   BBox and polygon drawing tools, drag/select/pan tool, a full undo/redo
   command stack (add/delete/move/resize/relabel), and 30-second autosave.

5. **Zero-shot bootstrap annotation** — `backend/app/ml/hf_zero_shot.py`
   Text-prompted, open-vocabulary detection (OWL-ViT via HuggingFace) to
   auto-annotate images before enough reviewed data exists to train a
   custom model.

6. **Custom model training** — `backend/app/ml/training_manager.py`,
   `backend/app/ml/custom_model_adapter.py`, `frontend/ui/train_model_dialog.py`
   Versioned model training (warm-starts from the previous version's
   weights), with a **graceful stop/cancel**: a SIGINT to the training
   subprocess finishes the current epoch and saves a checkpoint before
   exiting, so cancelling never loses in-progress work. Training progress
   streams live to the UI, and the training dialog can reattach to an
   already-running job if reopened mid-run.

7. **Auto-annotate loop** — `backend/app/api/autoannotate.py`, `frontend/ui/auto_annotate_dialog.py`
   Run the latest trained custom model on a single image or in batch across
   every unannotated image in a project.

8. **Multi-format export** — `backend/app/api/export.py`, `frontend/ui/export_dialog.py`
   Export a project as COCO JSON, YOLO TXT, or Pascal VOC XML, or as a full
   zip bundle (images + COCO annotations + class list).

9. **Storage abstraction** — `backend/app/core/storage/*.py`
   Local-disk storage for dev, pluggable cloud storage backend for
   production, selected by project `storage_mode`.

### Bug fixes (documented in code comments)

1. **SQLite lock contention during training** — `backend/app/db.py`
   A long-running training job holding a DB session open collided with
   concurrent log-progress writes and status polls, throwing "database is
   locked" — and those failures were being silently swallowed elsewhere.
   Fixed by enabling WAL journal mode plus a 30s busy-timeout at the
   connection level, so concurrent readers/writers no longer fight over a
   single lock.

2. **Training looked frozen in the UI** — `backend/app/api/autoannotate.py`
   Progress-log writes to the DB happened only every 10 lines, and a failed
   write was silently dropped — combined with the lock contention above, a
   perfectly healthy training run could look completely stalled with no
   error surfaced. Fixed to write every 2 lines and to log failures instead
   of swallowing them.

3. **Final training-job status silently failed to save** — `backend/app/api/autoannotate.py`
   The job's completion status was being committed through a SQLAlchemy
   object from a session that had already been closed (to free the DB
   during the long subprocess run) — committing a detached object persists
   nothing, with no error. Fixed by re-querying the job row on a fresh
   session before the final write.

4. **Training pulled in unused placeholder classes** — `backend/app/ml/training_manager.py`
   Dataset prep used to include every `LabelClass` row defined for a
   project, even leftover/placeholder ones (e.g. "class_1") nobody had
   actually annotated with. Fixed to derive the training class list only
   from classes that appear on at least one reviewed annotation.

5. **Layers panel could wipe canvas selection** — `frontend/ui/main_window.py`
   Rebuilding the layers list (`QListWidget.clear()` + re-`addItem()`) fires
   `itemSelectionChanged`, which would otherwise cascade back into the
   canvas and clear the very selection the list was just trying to mirror.
   Fixed by blocking signals on the layers list during rebuild.

### Known gaps (not yet fixed)

- ~~No way to mark an auto-annotation "reviewed" from the UI.~~ **Fixed
  2026-07-20** — see that date's feature #1.
