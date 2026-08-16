# Encrypted shared response cache

The cache remains shared across users: an identical analysis request can reuse
a paid model response. It never uses a user ID. Its SQLite values are encrypted
and its keys are HMACs, so a copied database does not reveal text or allow a
dictionary attack on common phrases.

## Required production secrets

Set two independent secrets before deploying the cache change. Generate them
locally; never place their values in Git, logs, or chat:

```powershell
$cacheEncryption = python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
$cacheHmac = python -c "import secrets; print(secrets.token_urlsafe(32))"
& "$env:USERPROFILE\.fly\bin\flyctl.exe" secrets set CACHE_ENCRYPTION_KEY=$cacheEncryption CACHE_HMAC_KEY=$cacheHmac --app speech-correction
```

Optional `RESPONSE_CACHE_MAX_ENTRIES` defaults to `5000`. The TTL is currently
30 days. The first encrypted startup deletes legacy plaintext cache rows; this
is intentional and only causes temporary cache misses.

When the analysis model, prompt policy, or output contract changes, bump
`ANALYSIS_PROMPT_VERSION` or `ANALYSIS_SCHEMA_VERSION`. Old results will not be
served to new requests.
