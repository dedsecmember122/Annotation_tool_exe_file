# AnnotationTool — Installation & User Guide

**Prepared by:** INSISO TECHNOLOGIES PRIVATE LIMITED
**Prepared for:** Bosch Rexroth
**Document version:** 1.0
**Date:** 2026-07-20

---

## Confidentiality Notice

This document and the accompanying software are provided to Bosch Rexroth
under the terms of the applicable agreement between Bosch Rexroth and
INSISO TECHNOLOGIES PRIVATE LIMITED. The software is proprietary; see the
`LICENSE` file distributed with it. This guide covers installation and
day-to-day use only — it does not describe internal implementation details.

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [First-Time Setup](#4-first-time-setup)
5. [Using the Application](#5-using-the-application)
6. [Auto-Annotation Workflow](#6-auto-annotation-workflow)
7. [Exporting Your Data](#7-exporting-your-data)
8. [Keyboard Shortcuts](#8-keyboard-shortcuts)
9. [Troubleshooting](#9-troubleshooting)
10. [Support](#10-support)

---

## 1. Overview

AnnotationTool is a self-contained, offline-capable desktop application for
labeling images (bounding boxes and polygons) and training a custom object
detection model on those labels. It runs entirely on the local machine —
no internet connection or external service is required for day-to-day use.

The application trains on **whatever object classes you define** for your
project. There is no fixed or built-in class list — you decide what the
model learns to detect.

---

## 2. System Requirements

| Requirement | Minimum |
|---|---|
| OS | Windows 10 or 11 (64-bit) |
| Disk space (application) | ~1 GB free (covers the 252 MB executable plus its temporary self-extraction on each launch) |
| Disk space (working data) | Variable — see note below |
| RAM | 8 GB (16 GB recommended if training larger models or datasets) |
| CPU | Any modern 64-bit x86 CPU. A GPU is not required — the shipped build runs on CPU. |
| Network | None required for normal use. The application only listens on `127.0.0.1` (localhost) — it does not open any port to the network. |

> **Working data sizing:** beyond the application itself, disk use depends
> on your image datasets (uploaded images + storage) and how many trained
> model checkpoints you keep. Each training run saves two checkpoint files
> (`last.pt` and `best.pt`); depending on the model size selected, a single
> checkpoint ranges from roughly 20 MB (smallest) to ~450 MB (largest).
> Plan disk space accordingly for your expected dataset size and how many
> project versions you intend to retain.

---

## 3. Installation

Two ways to install, depending on what you were given:

**Option A — Installer (`AnnotationToolSetup.exe`), recommended**
1. Double-click `AnnotationToolSetup.exe` and follow the wizard (license →
   install location → Start Menu/desktop shortcut → install).
2. No admin rights needed — it installs per-user under
   `%LocalAppData%\Programs\AnnotationTool`.
3. Launch from the Start Menu or desktop shortcut. To remove it later, use
   **Settings → Apps** (or the shortcut in its Start Menu folder) like any
   normal Windows application.

**Option B — Standalone executable (`AnnotationTool.exe`), portable**
1. Copy `AnnotationTool.exe` to the target machine (e.g. `C:\Program Files\AnnotationTool\` or any folder of your choice).
2. Double-click `AnnotationTool.exe` to launch. No installation step at all — it is a single, self-contained executable, useful for a quick test without leaving Start Menu entries behind.

### Windows SmartScreen Warning

Because this build is not yet signed with a commercial code-signing
certificate, Windows SmartScreen may show a warning on first launch of
either `AnnotationToolSetup.exe` or `AnnotationTool.exe`
("Windows protected your PC"). This is expected for an unsigned file, not a
sign of a problem with it. To proceed:

1. Click **More info**.
2. Click **Run anyway**.

If your organization's IT policy blocks unsigned executables outright,
please have IT allow-list `AnnotationTool.exe` (or contact INSISO
Technologies to arrange code signing before wider rollout).

---

## 4. First-Time Setup

On first launch, the application automatically creates its local data
files — there is nothing to configure manually for standard use:

| Item | Location |
|---|---|
| Database | `%USERPROFILE%\AnnotationTool\data\annotation_tool.db` |
| Uploaded images / storage | `%USERPROFILE%\AnnotationTool\storage\` |
| Trained models | `%USERPROFILE%\AnnotationTool\models\<project>\v<N>\best.pt` |
| Embedded backend | `http://127.0.0.1:8765` (localhost only) |

**The first account you create is automatically an administrator.**
Subsequent accounts are regular users. Use the in-app **Sign Up** screen to
create your first account, then log in.

No further configuration is required for local, single-machine use. If
your deployment requires custom settings (e.g. a shared network location
for storage, or a different port), contact INSISO Technologies — these are
set via an `.env` configuration file shipped alongside the executable and
should not be edited without guidance, since they affect where your data is
stored.

---

## 5. Using the Application

### 5.1 Create a Project

From the dashboard, create a new project and define the **label classes**
you want the model to learn (e.g. `component_a`, `defect_scratch`,
`missing_screw` — any names relevant to your use case). There is no
default or preset class list.

### 5.2 Upload Images

Upload the images you want to annotate into the project.

### 5.3 Annotate

Use the **Bounding Box** or **Polygon** tool to draw labels on each image,
then assign the correct class. Annotations can be edited, undone/redone,
and must be marked **reviewed** once you're satisfied with them (this is
what makes them count toward training).

See [Section 8](#8-keyboard-shortcuts) for the full shortcut list.

---

## 6. Auto-Annotation Workflow

This is the core workflow for training your own detection model and using
it to speed up labeling the rest of your dataset:

1. **Manually annotate at least 50 images** per project and mark them
   reviewed. (This minimum is enforced by the application — auto-annotate
   and training are unavailable below this count.)
2. Open **Auto-Annotate → Auto-Annotation Loop**.
3. Choose the number of **epochs** (recommended: 50–70 for a first pass)
   and the **train/validation split** (e.g. 80% train / 20% validation —
   you choose the percentage).
4. Click **Train Model**. Training runs in the background; a live log
   streams into the dialog. The best-performing checkpoint is kept
   automatically as `best.pt`.
5. Click **Save best.pt to Computer…** if you want to export the trained
   weights to a location of your choice.
6. Click **Auto-Annotate All Unannotated Images** to let the trained model
   label the remainder of your dataset automatically.
7. **Review** the auto-generated annotations, correct any mistakes, mark
   them reviewed, and optionally retrain — each additional round of
   review + retrain typically improves accuracy.

There is no fixed limit on how many times you repeat this loop.

---

## 7. Exporting Your Data

| Format | Output | Notes |
|---|---|---|
| COCO JSON | `.json` | Standard format for most ML frameworks |
| YOLO TXT | `.zip` of `.txt` files | One file per image, plus `classes.txt` |
| Pascal VOC XML | `.zip` of `.xml` files | One file per image |
| Full Dataset ZIP | `.zip` | Images + COCO annotations + class list, all together |

Exports are available from the project dashboard's **Export** option.

---

## 8. Keyboard Shortcuts

| Key | Action |
|---|---|
| `B` | Bounding box tool |
| `P` | Polygon tool |
| `Esc` | Select / pan tool |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Delete` | Delete selected annotation |
| `Ctrl+S` | Save annotations |
| `Ctrl+T` | Toggle dark / light theme |
| `Ctrl+F` | Fit image to view |
| Mouse wheel | Zoom in / out |
| Double-click | Close polygon (polygon mode) |
| `Enter` | Close polygon (polygon mode) |
| `Backspace` | Remove last polygon vertex |

---

## 9. Troubleshooting

**"Windows protected your PC" on launch**
Expected for an unsigned executable — see [Section 3](#3-installation).

**Auto-Annotate / Train Model options are greyed out**
You need at least 50 reviewed annotations in the project first.

**Application won't start / closes immediately**
Confirm you have at least 3 GB free disk space (the application creates a
local database and storage folder on first launch) and that antivirus
software has not quarantined the executable. Check
`%USERPROFILE%\AnnotationTool\` for a log file and share it with support
if the issue persists.

**"Startup Error: The embedded backend failed to start"**
On first launch, the app can take longer than usual to start (a fresh
antivirus scan of the executable, or normal cold-start on an unfamiliar
machine) — this dialog allows up to 90 seconds before giving up. If it
still appears, check `%USERPROFILE%\AnnotationTool\backend_error.log` —
if that file exists and has content, it contains the exact error and
should be shared with support. If the file is empty or missing, another
process may already be using port 8765; closing other running instances
of the app (check Task Manager) and relaunching usually resolves it.

**Training is slow**
Training runs on CPU in this build. Larger datasets or higher epoch counts
will take proportionally longer; there is no fixed time estimate as it
depends on your hardware and dataset size.

---

## 10. Support

For installation issues, licensing questions, or feature requests, contact
INSISO Technologies at **contact@insisotech.com**.

---

*© 2026 INSISO TECHNOLOGIES PRIVATE LIMITED. All rights reserved.*
