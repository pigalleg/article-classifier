# DVC and Google Drive Setup

This project uses DVC for selected generated result folders. Payloads are stored
in a private Google Drive remote named `gdrive`; DVC metadata is committed to
Git. Canonical calibration exports remain ordinary Git files.

## What Is Tracked

DVC tracks:

- selected run folders in `data/results/affinity_benchmark/`, each with its own
  `.dvc` manifest;
- `data/results/ensemble_weight_optimizer/`.

See `data/data-registry.yaml` for the complete target list. Other local
benchmark runs are intentionally outside DVC. The historical
`data/processed/affinity_calibration/to delete/` subtree is ignored by Git and
is not DVC-tracked.

## Install DVC

Use the project analysis environment:

```bash
~/.virtualenvs/classifier/bin/python -m pip install "dvc[gdrive]"
~/.virtualenvs/classifier/bin/dvc version
```

## Remote Configuration

The shared configuration in `.dvc/config` defines the default remote. It must
use the Google Drive scheme:

```bash
~/.virtualenvs/classifier/bin/dvc remote list
```

Expected form:

```text
gdrive    gdrive://FOLDER_ID
```

Never commit private OAuth values. `.dvc/config.local` is ignored by
`.dvc/.gitignore`.

If the remote needs to be changed:

```bash
~/.virtualenvs/classifier/bin/dvc remote modify gdrive url "gdrive://FOLDER_ID"
```

## Google OAuth

If Google blocks the shared DVC application, create a Desktop OAuth client in
Google Cloud, enable the Google Drive API, and add the Google account as a test
user while the consent screen is in Testing. Configure the client locally:

```bash
~/.virtualenvs/classifier/bin/dvc remote modify --local gdrive gdrive_client_id "YOUR_CLIENT_ID"
~/.virtualenvs/classifier/bin/dvc remote modify --local gdrive gdrive_client_secret "YOUR_CLIENT_SECRET"
```

Store the generated user credential outside the repository:

```bash
mkdir -p "$HOME/.config/dvc/article-classifier"
chmod 700 "$HOME/.config/dvc/article-classifier"

~/.virtualenvs/classifier/bin/dvc remote modify --local gdrive \
  gdrive_user_credentials_file "$HOME/.config/dvc/article-classifier/gdrive-auth.json"
```

The first remote operation may print an OAuth URL. Open it manually if the
terminal cannot launch a browser. After authentication:

```bash
chmod 600 "$HOME/.config/dvc/article-classifier/gdrive-auth.json"
```

## Push and Pull

Push selected result targets:

```bash
~/.virtualenvs/classifier/bin/dvc push -r gdrive
```

Verify the local cache and remote agree:

```bash
~/.virtualenvs/classifier/bin/dvc status -c
```

On another machine, clone the repository, configure local OAuth credentials,
and retrieve the selected results:

```bash
git pull
~/.virtualenvs/classifier/bin/dvc pull -r gdrive
```

## Updating a Result Target

After changing a tracked result folder, refresh its individual manifest, commit
only metadata, then push:

```bash
~/.virtualenvs/classifier/bin/dvc add data/results/affinity_benchmark/RUN_ID
git add data/results/affinity_benchmark/RUN_ID.dvc data/results/affinity_benchmark/.gitignore
git commit -m "Update benchmark run RUN_ID"
~/.virtualenvs/classifier/bin/dvc push -r gdrive
```

Do not run `git add data/results/affinity_benchmark/RUN_ID/`; the actual result
payload is DVC-managed.

## Compatibility Issue

If a remote operation fails with `module 'lib' has no attribute 'GEN_EMAIL'`,
the Python OpenSSL packages are incompatible. In the classifier environment,
the compatible package set is:

```bash
~/.virtualenvs/classifier/bin/python -m pip install \
  "PyDrive2==1.21.2" \
  "pyOpenSSL==24.2.1" \
  "cryptography==43.0.3" \
  "asyncssh==2.23.1"
```

Then check the environment and remote:

```bash
~/.virtualenvs/classifier/bin/python -m pip check
~/.virtualenvs/classifier/bin/dvc status -c
```

## Security Checks

Before committing DVC metadata:

```bash
git status
```

Do not stage `.dvc/config.local` or `$HOME/.config/dvc/article-classifier/gdrive-auth.json`.
