# Firebase backend authentication migration

## Purpose

New app versions authenticate to the FastAPI server with a Firebase ID token.
The server verifies the token signature and uses the verified Firebase UID for
rate limiting. This replaces the static `X-API-Key` embedded in old APK/IPA
files, which must be treated as public.

## Deploy order

1. In Google Cloud Console for Firebase project `speechcorrection-4118e`, create
   a service account key with only the permissions required by Firebase Admin
   token verification. Download the JSON privately; never commit it.
2. On the production Fly app, add the JSON as a single secret:

   ```powershell
   $serviceAccountPath = 'C:\secure\speechcorrection-firebase-admin.json'
   $serviceAccountJson = [IO.File]::ReadAllText($serviceAccountPath)
   flyctl secrets set FIREBASE_SERVICE_ACCOUNT_JSON="$serviceAccountJson" --app speech-correction
   flyctl secrets set FIREBASE_PROJECT_ID=speechcorrection-4118e --app speech-correction
   ```

   Repeat for `speech-correction-dev` if the development app must accept real
   mobile clients.
3. Deploy this server version with `ALLOW_LEGACY_API_KEY=true`. Confirm `/health`
   is healthy and a new app build can analyse text and open a rule.
4. Release the Flutter client that sends Firebase Bearer tokens. Keep legacy
   auth enabled until the old public releases no longer matter operationally.
5. Set `ALLOW_LEGACY_API_KEY=false`, deploy, then rotate/remove `API_KEY` from
   Fly. Verify a current client works and an old `X-API-Key` request receives 401.

## App Check (next layer)

After Firebase ID token auth is live, enable Firebase App Check in **monitoring**
mode first:

- Android: Play Integrity;
- iOS: App Attest (or DeviceCheck where App Attest is unavailable);
- server: verify the App Check token in addition to the Firebase ID token.

Review the monitoring metrics for at least one release cycle before enforcing.
Authentication identifies a user; App Check makes it harder for modified apps
and scripts to impersonate the official client. They complement each other.

### Rollout status — 2026-08-16

- Android app `com.daniilnovykov.speech_correction` is registered with **Play
  Integrity** in Firebase App Check.
- The server verifies the `X-Firebase-AppCheck` header when it is present but
  accepts missing or invalid tokens while `APP_CHECK_ENFORCED=false` (the
  default). This is the monitoring phase for the custom Fly backend.
- iOS registration remains pending. Firebase requires an Apple Developer
  authentication key (`.p8`), its Key ID and Apple Team ID to register
  DeviceCheck/App Attest. Do not enable enforcement until iOS is registered,
  the new app builds are live and monitoring shows valid tokens.
- After 1–2 weeks of monitoring, review Fly request logs (the `app_check=` field)
  and App Check metrics. If
  legitimate current clients consistently send valid tokens, set
  `APP_CHECK_ENFORCED=true`, deploy, and verify current Android and iOS builds
  before turning it on permanently.

## Incident response

The historical static client key is public by design once distributed. It should
not be reused for any new secret. If a server, DeepSeek or Firebase service-account
secret is exposed, rotate that secret in its provider console, update the Fly
secret, redeploy, and inspect access/cost logs. Do not paste secrets into source,
issue trackers, chat or release notes.
