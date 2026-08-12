; installer.iss — wraps the PyInstaller one-folder build into a single
; AnyaTennis-Setup-<version>.exe for beta testers.
;
; This is the Windows counterpart to make_dmg.sh: testers get one file to
; download and double-click, a Start Menu entry, and a working uninstaller.
;
; Do NOT run this directly — it needs both version strings passed in, so the
; installer can never drift from desktop/version.py. build_windows.ps1 reads
; APP_VERSION and invokes:
;
;   iscc /DAppVersion=0.1.0-beta.3 /DVersionNumeric=0.1.0.0 installer.iss
;
; Two forms are needed because they go to different places: AppVersion is the
; human-readable string shown in the wizard and Add/Remove Programs, while
; VersionInfoVersion writes the Win32 version resource, which is four integers
; and rejects a "-beta.3" suffix outright.
;
; Requires Inno Setup 6.3+ (winget install JRSoftware.InnoSetup) — earlier
; releases don't understand the "x64compatible" architecture identifier.

#ifndef AppVersion
  #error AppVersion is not defined — run this via build_windows.ps1, not directly.
#endif
#ifndef VersionNumeric
  #error VersionNumeric is not defined — run this via build_windows.ps1, not directly.
#endif

#define AppName "Anya Tennis"
#define AppExeName "AnyaTennis.exe"
#define AppPublisher "Anya Tennis"

[Setup]
; A stable GUID is what lets an upgrade replace the previous install instead
; of stacking a second copy in Add/Remove Programs. Never regenerate it.
AppId={{8E4B1C2A-6F3D-4A7E-9B15-2C8D5E7A9F31}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
VersionInfoVersion={#VersionNumeric}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
; Relative paths here resolve against this .iss file's own directory
; (desktop/), which is also where PyInstaller writes dist\ when the spec is
; run from desktop/. So both the payload and the finished installer sit in
; desktop\dist\ alongside each other.
OutputDir=dist
OutputBaseFilename=AnyaTennis-Setup-{#AppVersion}
SetupIconFile=assets\icon\AnyaTennis.ico
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} {#AppVersion}
WizardStyle=modern
DisableProgramGroupPage=yes

; lowest => install for the current user only, into %LOCALAPPDATA%\Programs,
; with no UAC prompt. Deliberate: the build is unsigned, so an elevation
; prompt for an unrecognised publisher is exactly the dialog that makes a
; beta tester abandon the install. Nothing here needs machine-wide access.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; The payload is ~2 GB of mostly-incompressible weights and DLLs. LZMA2 at
; max compression turns the build into a very long, very hot CI step for a
; few percent; solid compression on this many files also makes the installer
; slow to start. Defaults are the right trade here.
Compression=lzma2/fast
SolidCompression=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The whole PyInstaller one-folder output. recursesubdirs+createallsubdirs
; keeps the internal layout byte-for-byte, which matters because PyInstaller
; resolves everything relative to the exe's _internal directory.
Source: "dist\AnyaTennis\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; PyInstaller writes nothing outside {app}, but Python leaves __pycache__
; directories behind that the uninstaller's file manifest doesn't know about,
; which would otherwise strand {app} on disk after an uninstall.
Type: filesandordirs; Name: "{app}\_internal"
Type: dirifempty; Name: "{app}"
