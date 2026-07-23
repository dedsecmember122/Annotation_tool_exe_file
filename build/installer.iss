; Inno Setup script for AnnotationTool.
; Wraps dist/AnnotationTool.exe (built by PyInstaller, see annotation_tool.spec)
; in a standard Windows installer wizard: license, install location, Start Menu
; + optional desktop shortcut, and a proper uninstaller entry.
;
; Build locally (requires Inno Setup — https://jrsoftware.org/isinfo.php):
;   iscc build\installer.iss
; Output: dist_installer\AnnotationToolSetup.exe
;
; CI builds this automatically after the PyInstaller step — see
; .github/workflows/build-windows.yml.

; MyAppExeName intentionally stays "AnnotationTool.exe" — that's the
; internal PyInstaller output filename (annotation_tool.spec); only the
; user-facing name (shortcuts, install folder, wizard text) is rebranded.
#define MyAppName "InSiSo Model Bench"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "InSiSo Technologies Private Limited"
#define MyAppExeName "AnnotationTool.exe"

[Setup]
AppId={{08661929-DD30-4C17-BDCA-4AF9E53CAFCE}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Per-user install under %LocalAppData% — no admin rights / UAC prompt
; needed, so it works on locked-down corporate machines too.
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist_installer
OutputBaseFilename=AnnotationToolSetup
Compression=lzma2
SolidCompression=yes
SetupIconFile=..\frontend\resources\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"

[Files]
Source: "..\dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
