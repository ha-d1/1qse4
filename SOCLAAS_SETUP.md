# SoCLaaS Provider Setup

This supplemental guide does not replace or alter the hackathon requirements in `README.md`.
The benchmark remains within-user ranking with `long_view`, GAUC, nDCG@5, and their mean.
Development uses train and validation labels only; hidden-test labels must not be evaluated.

## Local setup

Install the declared dependencies into a local virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Enter the SoCLaaS key without displaying it or putting it into shell history:

```bash
read -s "SOCLAAS_API_KEY?SoCLaaS API key: "
echo
export SOCLAAS_API_KEY
export SOCLAAS_BASE_URL="https://soclaas-api.comp.nus.edu.sg/v1"
```

Check connectivity and model availability. This prints model IDs, never the key:

```bash
.venv/bin/python soclaas_check.py --model qwen3-coder-next
```

Run one bounded development iteration, pointing to the local dataset if it is outside this checkout:

```bash
.venv/bin/python agent.py \
  --max-iterations 1 \
  --data-dir "/path/to/KuaiRand-Pure/data"
```

Remove the credential from the current shell when finished:

```bash
unset SOCLAAS_API_KEY
```

## Optional: automate future runs with macOS Keychain

Store the key once. Put `-w` last so macOS prompts for the value instead of placing it in
shell history:

```bash
security add-generic-password \
  -a "$USER" \
  -s "TikTok-TechJam-SoCLaaS" \
  -U \
  -w
```

After that, one validation-only iteration requires no key entry or data-path argument:

```bash
.venv/bin/python run_agent_with_keychain.py
```

Remove the saved credential when it is no longer needed:

```bash
security delete-generic-password -a "$USER" -s "TikTok-TechJam-SoCLaaS"
```
