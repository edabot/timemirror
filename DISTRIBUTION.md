# Distribution Guide

How to build, sign, notarize, and distribute TimeMirror.app.

## Prerequisites (one-time setup)

### 1. Apple Developer Program
Enroll at [developer.apple.com](https://developer.apple.com) ($99/year). Required for a Developer ID Application certificate and notarization.

### 2. Developer ID Application certificate
- Go to [developer.apple.com/account/resources/certificates/add](https://developer.apple.com/account/resources/certificates/add)
- Choose **Developer ID Application**
- Generate a CSR via Keychain Access → Certificate Assistant → Request a Certificate from a CA (save to disk)
- Upload the `.certSigningRequest`, download the `.cer`, double-click to install

### 3. Store notarization credentials (one-time)
Generate an app-specific password at [appleid.apple.com](https://appleid.apple.com) → Security → App-Specific Passwords, then:

```bash
make notarize-setup
```

Enter the app-specific password when prompted. Credentials are stored securely in Keychain under the profile name `timemirror` and never need to be entered again.

---

## Building a release

```bash
make clean && make notarize
```

This runs the full pipeline:

| Step | Target | Output |
|------|--------|--------|
| Compile binary | `time_mirror` | `./time_mirror` |
| Bundle dylibs | `dist` | `dist/libs/` + `time_mirror_mac.zip` |
| Build app bundle | `app` | `TimeMirror.app` |
| Zip for submission | `app-zip` | `TimeMirror.zip` |
| Submit to Apple + staple | `notarize` | `TimeMirror_notarized.zip` |

The final artifact is **`TimeMirror_notarized.zip`**.

---

## Distribution

Send recipients **`TimeMirror_notarized.zip`**. Include these instructions:

1. Unzip
2. **Drag `TimeMirror.app` to `/Applications`** before opening
3. Double-click to open — grant camera access when prompted
4. The app retries camera access for ~10 seconds while the permission dialog is open

> **Critical:** Opening the app directly from a zip or Downloads folder triggers macOS App Translocation — the app runs from a randomised temp path and will fail to launch. It must be in `/Applications` (or another permanent location).

---

## Signing details

| Item | Value |
|------|-------|
| Certificate | `Developer ID Application: Edward Lewis (Z9B4288ZRX)` |
| Bundle ID | `com.edlewis.timemirror` |
| Team ID | `Z9B4288ZRX` |
| Entitlements | `com.apple.security.device.camera` |
| Hardened runtime | Yes (`--options runtime`) |
| Notarization profile | `timemirror` (Keychain) |

---

## Troubleshooting

**`bundle format unrecognized, invalid, or unsuitable`**
- `Info.plist` must exist before `codesign` runs. The Makefile writes it first now — if this error reappears, check the `app` target ordering.

**`HTTP 401` from notarytool**
- The stored app-specific password may have expired or been revoked. Generate a new one at [appleid.apple.com](https://appleid.apple.com) and re-run `make notarize-setup`.

**`duplicate LC_RPATH` crash on other machine**
- The `dist` target deduplicates LC_RPATH entries after `dylibbundler`. If this reappears after changing dylib deps, run `make clean && make notarize`.

**Gatekeeper warning on recipient's machine**
- App must be notarized AND stapled. `make notarize` does both. If they still see a warning, check that they're opening from `/Applications`, not from the zip.

**Camera denied on first launch**
- macOS shows the permission dialog but the old code exited before the user could approve. The current code retries for 10 seconds. If it still fails, the user can reset permissions with:
  ```bash
  tccutil reset Camera com.edlewis.timemirror
  ```
  then reopen the app.
