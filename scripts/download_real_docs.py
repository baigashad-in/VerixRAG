"""
Download real public documents for RAG evaluation.

Pulls genuine human-written documents from:
1. GitHub repos — contributing guides, security policies, READMEs, docs
2. Public government/institutional sources
3. Open-source project documentation

These are messy, real, and diverse — exactly what RAG needs to handle.
"""

import os
import re
import json
import time
import argparse
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import requests

# ─── DOCUMENT SOURCES ───
# Each entry: (filename, raw URL, domain/category)
# All sources are public, openly licensed (MIT/Apache/CC/public domain)

GITHUB_DOCS = [
    # ── Contributing Guides (real dev processes) ──
    ("contrib_react.md",
     "https://raw.githubusercontent.com/facebook/react/main/CONTRIBUTING.md",
     "open_source"),
    ("contrib_vscode.md",
     "https://raw.githubusercontent.com/microsoft/vscode/main/CONTRIBUTING.md",
     "open_source"),
    ("contrib_rust.md",
     "https://raw.githubusercontent.com/rust-lang/rust/master/CONTRIBUTING.md",
     "open_source"),
    ("contrib_tensorflow.md",
     "https://raw.githubusercontent.com/tensorflow/tensorflow/master/CONTRIBUTING.md",
     "open_source"),
    ("contrib_django.md",
     "https://raw.githubusercontent.com/django/django/main/CONTRIBUTING.rst",
     "open_source"),
    ("contrib_kubernetes.md",
     "https://raw.githubusercontent.com/kubernetes/community/master/contributors/guide/README.md",
     "open_source"),
    ("contrib_nodejs.md",
     "https://raw.githubusercontent.com/nodejs/node/main/CONTRIBUTING.md",
     "open_source"),
    ("contrib_flutter.md",
     "https://raw.githubusercontent.com/flutter/flutter/master/CONTRIBUTING.md",
     "open_source"),

    # ── Security Policies (real incident/vuln processes) ──
    ("security_electron.md",
     "https://raw.githubusercontent.com/electron/electron/main/SECURITY.md",
     "security"),
    ("security_nextjs.md",
     "https://raw.githubusercontent.com/vercel/next.js/canary/SECURITY.md",
     "security"),
    ("security_django.md",
     "https://raw.githubusercontent.com/django/django/main/docs/internals/security.txt",
     "security"),
    ("security_curl.md",
     "https://raw.githubusercontent.com/curl/curl/master/docs/SECURITY-ADVISORY.md",
     "security"),
    ("security_kubernetes.md",
     "https://raw.githubusercontent.com/kubernetes/community/master/committee-security-response/README.md",
     "security"),

    # ── Codes of Conduct (HR/community policies) ──
    ("coc_contributor_covenant.md",
     "https://raw.githubusercontent.com/EthicalSource/contributor_covenant/release/content/version/2/1/code_of_conduct.md",
     "policy"),
    ("coc_golang.md",
     "https://raw.githubusercontent.com/golang/go/master/CONTRIBUTING.md",
     "policy"),
    ("coc_rust.md",
     "https://raw.githubusercontent.com/rust-lang/www.rust-lang.org/master/static/policies/code-of-conduct.md",
     "policy"),

    # ── Technical Documentation (real product docs) ──
    ("docs_fastapi_tutorial.md",
     "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/first-steps.md",
     "technical"),
    ("docs_fastapi_security.md",
     "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/security/index.md",
     "technical"),
    ("docs_docker_getstarted.md",
     "https://raw.githubusercontent.com/docker/docs/main/content/get-started/introduction/develop-with-containers.md",
     "technical"),
    ("docs_postgres_backup.md",
     "https://raw.githubusercontent.com/postgres/postgres/master/doc/src/sgml/backup.sgml",
     "technical"),
    ("docs_redis_security.md",
     "https://raw.githubusercontent.com/redis/redis-doc/master/docs/management/security/index.md",
     "technical"),
    ("docs_nginx_beginners.md",
     "https://raw.githubusercontent.com/nginx/nginx/master/docs/beginners_guide",
     "technical"),

    # ── Project READMEs (varied structure and quality) ──
    ("readme_htmx.md",
     "https://raw.githubusercontent.com/bigskysoftware/htmx/master/README.md",
     "technical"),
    ("readme_fastapi.md",
     "https://raw.githubusercontent.com/fastapi/fastapi/master/README.md",
     "technical"),
    ("readme_langchain.md",
     "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md",
     "technical"),
    ("readme_ollama.md",
     "https://raw.githubusercontent.com/ollama/ollama/main/README.md",
     "technical"),
    ("readme_deno.md",
     "https://raw.githubusercontent.com/denoland/deno/main/README.md",
     "technical"),
    ("readme_pytorch.md",
     "https://raw.githubusercontent.com/pytorch/pytorch/main/README.md",
     "technical"),
    ("readme_homebrew.md",
     "https://raw.githubusercontent.com/Homebrew/brew/master/README.md",
     "technical"),
    ("readme_svelte.md",
     "https://raw.githubusercontent.com/sveltejs/svelte/main/README.md",
     "technical"),

    # ── Governance / Process docs ──
    ("governance_nodejs.md",
     "https://raw.githubusercontent.com/nodejs/node/main/GOVERNANCE.md",
     "governance"),
    ("governance_rust.md",
     "https://raw.githubusercontent.com/rust-lang/rfcs/master/text/1068-rust-governance.md",
     "governance"),
    ("changelog_vscode.md",
     "https://raw.githubusercontent.com/microsoft/vscode/main/CHANGELOG.md",
     "changelog"),

    # ── License files (legal documents) ──
    ("license_apache2.md",
     "https://raw.githubusercontent.com/apache/.github/main/LICENSE",
     "legal"),
    ("license_mit_babel.md",
     "https://raw.githubusercontent.com/babel/babel/main/LICENSE",
     "legal"),
    ("license_gpl3.md",
     "https://raw.githubusercontent.com/torvalds/linux/master/COPYING",
     "legal"),

    # ── Style guides (coding standards / real team docs) ──
    ("style_google_python.md",
     "https://raw.githubusercontent.com/google/styleguide/gh-pages/pyguide.md",
     "style_guide"),
    ("style_airbnb_js.md",
     "https://raw.githubusercontent.com/airbnb/javascript/master/README.md",
     "style_guide"),
    ("style_google_typescript.md",
     "https://raw.githubusercontent.com/google/gts/main/README.md",
     "style_guide"),

    # ── GitHub community health files ──
    ("support_vscode.md",
     "https://raw.githubusercontent.com/microsoft/vscode/main/SUPPORT.md",
     "support"),
    ("support_tensorflow.md",
     "https://raw.githubusercontent.com/tensorflow/tensorflow/master/ISSUES.md",
     "support"),

    # ── Ops / Deployment docs ──
    ("ops_gitops_flux.md",
     "https://raw.githubusercontent.com/fluxcd/flux2/main/README.md",
     "devops"),
    ("ops_terraform_getstarted.md",
     "https://raw.githubusercontent.com/hashicorp/terraform/main/README.md",
     "devops"),
    ("ops_ansible_readme.md",
     "https://raw.githubusercontent.com/ansible/ansible/devel/README.md",
     "devops"),

    # ── Data / ML docs ──
    ("ml_huggingface_readme.md",
     "https://raw.githubusercontent.com/huggingface/transformers/main/README.md",
     "ml"),
    ("ml_scikit_contrib.md",
     "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/CONTRIBUTING.md",
     "ml"),
    ("ml_pandas_readme.md",
     "https://raw.githubusercontent.com/pandas-dev/pandas/main/README.md",
     "ml"),
]

# Documents that need HTML-to-text conversion
PUBLIC_WEB_DOCS = [
    # These are fetched as HTML and converted to markdown-ish text
    ("faq_irs_filing.md",
     "https://www.irs.gov/faqs/filing-requirements-status-dependents/general",
     "government"),
    ("guide_ada_compliance.md",
     "https://www.ada.gov/resources/small-business/",
     "government"),
    ("guide_ftc_privacy.md",
     "https://www.ftc.gov/business-guidance/privacy-security/data-security",
     "government"),
]


def download_raw(url: str, timeout: int = 15) -> str | None:
    """Download a raw text file."""
    try:
        headers = {"User-Agent": "VerixRAG-Eval/1.0 (educational research)"}
        resp = requests.get(url, timeout=timeout, headers=headers)

        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        content = resp.text

        # Skip if too short (probably an error page)
        if len(content) < 200:
            return None

        return content

    except Exception as e:
        print(f" error: {e}")
        return None


def download_html_as_text(url: str, timeout: int = 15) -> str | None:
    """Download HTML page and extract text content."""
    try:
        headers = {"User-Agent": "VerixRAG-Eval/1.0 (educational research)"}
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()

        html = resp.text

        # Basic HTML to text (no BeautifulSoup dependency needed)
        # Remove scripts and styles
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)

        # Convert common elements
        text = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'\n## \1\n', text)
        text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL)
        text = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', text, flags=re.DOTALL)
        text = re.sub(r'<br\s*/?>', '\n', text)

        # Remove remaining tags
        text = re.sub(r'<[^>]+>', '', text)

        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()

        if len(text) < 200:
            return None

        return text

    except Exception as e:
        print(f" error: {e}")
        return None


def clean_document(content: str, source_url: str) -> str:
    """Clean up downloaded content for RAG ingestion."""

    # Truncate very large files (changelogs, etc.)
    MAX_CHARS = 15_000
    if len(content) > MAX_CHARS:
        # Try to cut at a section boundary
        truncated = content[:MAX_CHARS]
        last_heading = truncated.rfind('\n#')
        if last_heading > MAX_CHARS * 0.7:
            truncated = truncated[:last_heading]
        content = truncated + "\n\n[Document truncated for evaluation purposes]"

    # Add source attribution header
    domain = urlparse(source_url).netloc
    header = f"<!-- Source: {source_url} -->\n<!-- Domain: {domain} -->\n\n"

    return header + content


def main():
    parser = argparse.ArgumentParser(
        description="Download real public documents for RAG evaluation"
    )
    parser.add_argument("--output-dir", default="./documents",
                        help="Output directory")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between downloads")
    parser.add_argument("--skip-web", action="store_true",
                        help="Skip HTML web pages, only download raw files")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip files that already exist")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sources = GITHUB_DOCS.copy()
    if not args.skip_web:
        all_sources.extend(PUBLIC_WEB_DOCS)

    total = len(all_sources)
    downloaded = 0
    skipped = 0
    failed = 0
    failed_list = []

    print(f"{'='*60}")
    print(f"Downloading {total} real public documents")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Track domains for summary
    domains = {}

    for i, (filename, url, domain) in enumerate(all_sources):
        filepath = output_dir / filename
        domains[domain] = domains.get(domain, 0) + 1

        # Skip existing
        if args.skip_existing and filepath.exists():
            print(f"[{i+1}/{total}] SKIP: {filename}")
            skipped += 1
            continue

        print(f"[{i+1}/{total}] {filename} [{domain}]...", end="", flush=True)

        # Download
        if url.startswith("https://raw.githubusercontent.com") or \
           url.endswith(('.md', '.rst', '.txt', '.sgml')):
            content = download_raw(url)
        else:
            content = download_html_as_text(url)

        if content:
            cleaned = clean_document(content, url)
            filepath.write_text(cleaned, encoding="utf-8")
            word_count = len(cleaned.split())
            print(f" ✓ ({word_count} words)")
            downloaded += 1
        else:
            print(f" ✗ FAILED")
            failed += 1
            failed_list.append((filename, url))

        time.sleep(args.delay)

    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Downloaded:  {downloaded}")
    print(f"Skipped:     {skipped}")
    print(f"Failed:      {failed}")
    print(f"Total files: {len(list(output_dir.glob('*.md')))}")

    print(f"\nDomain breakdown:")
    for domain, count in sorted(domains.items()):
        existing = len(list(output_dir.glob(f"*{domain}*")))
        print(f"  {domain}: {count} sources")

    if failed_list:
        print(f"\nFailed downloads (may need manual download):")
        for filename, url in failed_list:
            print(f"  {filename}: {url}")

    # Save manifest
    manifest = {
        "downloaded": downloaded,
        "failed": failed,
        "total_files": len(list(output_dir.glob("*.md"))),
        "sources": [
            {"filename": f, "url": u, "domain": d}
            for f, u, d in all_sources
        ],
        "failed_urls": [
            {"filename": f, "url": u}
            for f, u in failed_list
        ],
    }

    manifest_path = output_dir / "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {manifest_path}")
    print(f"\nNext steps:")
    print(f"  1. python scripts/ingest_docs.py")
    print(f"  2. uvicorn src.api.main:app --reload")
    print(f"  3. python scripts/generate_eval_dataset.py --per-category 8")
    print(f"  4. python scripts/run_scaled_eval.py")


if __name__ == "__main__":
    main()