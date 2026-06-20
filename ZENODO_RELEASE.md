# Minting a Zenodo DOI for this repository

This repo ships a `.zenodo.json` so that a GitHub Release is automatically
archived to Zenodo with the correct metadata. **No DOI has been minted yet.**
Minting is a deliberate one-time human action — follow the steps below.

## One-time setup (only if not done before)

1. Sign in at https://zenodo.org with your GitHub account.
2. Go to Zenodo -> **Settings -> GitHub**.
3. Find `ChatchaiTritham/Causal-CDSS` in the list and flip the toggle **ON**.
   (Zenodo only archives releases created *after* the toggle is enabled.)

## Mint the DOI (the one step)

Create and push a version tag + GitHub Release. The simplest path:

```bash
git tag -a v1.0.0 -m "Causal-CDSS v1.0.0"
git push origin v1.0.0
```

Then on GitHub: **Releases -> Draft a new release -> choose tag `v1.0.0` ->
Publish release.** Within a minute Zenodo ingests the release, reads
`.zenodo.json`, and mints a DOI.

(Alternatively, do the whole thing on GitHub: **Releases -> Draft a new release**,
type `v1.0.0` as a new tag, target `main`, and Publish — GitHub creates the tag
for you.)

## After minting

1. Copy the version DOI badge/URL from the Zenodo record.
2. Add the DOI to `README.md` (badge + a "Data and code availability" line) and to
   the manuscript's data-availability statement.
3. The concept DOI (resolves to the latest version) is the one to cite in the paper.

## Notes

- Use semantic version tags (`v1.0.0`, `v1.1.0`, ...). Each new release gets a new
  version DOI under the same concept DOI.
- `.zenodo.json` is the source of truth for title, authors, ORCIDs, license, and
  keywords on the Zenodo record — edit it there, not in the Zenodo web form.
