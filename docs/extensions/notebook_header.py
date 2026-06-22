# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Sphinx extension: per-notebook downloadable zip + header link.

For every MyST-NB notebook page (``.md``/``.ipynb`` source under ``tutorials/``
or ``examples/``) this extension:

* injects a download link at the top of the rendered page (``doctree-read``),
  pointing at a zip that sits next to the page's HTML output, and
* builds that zip at the end of the build (``build-finished``). The zip contains
  a real ``.ipynb`` (converted from the MyST source with jupytext) plus the
  notebook's ``files/`` data directory, if present.

The only MyST construct that does not survive the ``.md -> .ipynb`` conversion is
the Sphinx ``{doc}`` cross-reference role; it is rewritten into an absolute
markdown link to the published HTML docs (using ``html_baseurl``). The source
``.md`` files are never modified -- rewriting happens on the in-memory notebook
that jupytext produces, before it is serialized into the zip.
"""

import posixpath
import re
import zipfile
from pathlib import Path

from docutils import nodes
from sphinx.util import logging

logger = logging.getLogger(__name__)

# Only act on pages whose docname lives under one of these prefixes ...
NOTEBOOK_DIRS = ("tutorials/", "examples/")
# ... and whose source file is an actual notebook (not the .rst overviews).
NOTEBOOK_SUFFIXES = (".md", ".ipynb")

# Matches {doc}`target` and {doc}`Title <target>`.
_DOC_ROLE = re.compile(r"\{doc\}`(?:([^`<]*?)\s*<)?([^`<>]+?)>?`")
# Matches a markdown link whose target is a sibling .md page: [text](path.md)
_MD_LINK = re.compile(r"(\]\()([^)]+?)\.md(#[^)]*)?(\))")


def _is_notebook(app, docname):
    if not docname.startswith(NOTEBOOK_DIRS):
        return False
    return Path(app.env.doc2path(docname)).suffix in NOTEBOOK_SUFFIXES


def _zip_basename(docname):
    """Basename used both for the artifact file and the top-level folder inside a zip."""
    return posixpath.basename(docname)


def _files_dir(app, docname):
    """Path to the notebook's sibling ``files/`` data directory (may not exist)."""
    return Path(app.env.doc2path(docname)).parent / "files"


def on_doctree_read(app, doctree):
    """Inject a 'download this notebook' link at the top of notebook pages."""
    docname = app.env.docname
    if not _is_notebook(app, docname):
        return
    refuri = _zip_basename(docname) + ".zip"
    para = nodes.paragraph(classes=["notebook-download"])
    para += nodes.reference(
        "", "Download this notebook (.ipynb + data files)", refuri=refuri
    )
    doctree.insert(0, para)


def _resolve_target(current_docname, target):
    """Resolve a {doc} target (possibly relative) to an absolute docname."""
    if target.startswith("/"):
        return target.lstrip("/")
    base_dir = posixpath.dirname(current_docname)
    return posixpath.normpath(posixpath.join(base_dir, target))


def _rewrite_xrefs(text, current_docname, app):
    """Rewrite {doc} roles (and sibling .md links) into absolute HTML links.

    Operates on a single markdown cell's source text and returns the rewritten
    text. Cross-references point at ``html_baseurl/<docname>.html``.
    """
    base = (app.config.html_baseurl or "").rstrip("/")

    def doc_repl(match):
        title, target = match.group(1), match.group(2).strip()
        target_docname = _resolve_target(current_docname, target)
        if not title:
            title_node = app.env.titles.get(target_docname)
            title = title_node.astext() if title_node is not None else posixpath.basename(target_docname)
        return f"[{title}]({base}/{target_docname}.html)"

    text = _DOC_ROLE.sub(doc_repl, text)

    def md_link_repl(match):
        target = match.group(2)
        if "://" in target:  # leave external URLs (e.g. .../README.md) untouched
            return match.group(0)
        anchor = match.group(3) or ""
        target_docname = _resolve_target(current_docname, target)
        return f"{match.group(1)}{base}/{target_docname}.html{anchor}{match.group(4)}"

    text = _MD_LINK.sub(md_link_repl, text)
    return text


def _build_notebook_zip(app, docname, jupytext, nbformat):
    """Convert one notebook to .ipynb and bundle it (plus files/) into a zip."""
    nb = jupytext.read(Path(app.env.doc2path(docname)))

    for cell in nb.cells:
        if cell.get("cell_type") == "markdown":
            cell["source"] = _rewrite_xrefs(cell["source"], docname, app)

    nb_json = nbformat.writes(nb)

    base = _zip_basename(docname)
    out_dir = Path(app.outdir).joinpath(*docname.split("/")).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_dir / f"{base}.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base}/{base}.ipynb", nb_json)
        files_dir = _files_dir(app, docname)
        if files_dir.is_dir():
            for path in sorted(files_dir.rglob("*")):
                if path.is_file():
                    arcname = f"{base}/files/{path.relative_to(files_dir).as_posix()}"
                    zf.write(path, arcname)


def on_build_finished(app, exception):
    """Build a downloadable zip for every notebook into the HTML output tree."""
    if exception is not None:
        return
    if app.builder.name not in ("html", "dirhtml"):
        return

    try:
        import jupytext
        import nbformat
    except ImportError as err:
        logger.warning(
            "notebook_header: jupytext/nbformat unavailable (%s); "
            "skipping notebook zip generation.", err
        )
        return

    count = 0
    for docname in sorted(app.env.found_docs):
        if not _is_notebook(app, docname):
            continue
        try:
            _build_notebook_zip(app, docname, jupytext, nbformat)
            count += 1
        except Exception as err:  # noqa: BLE001 - never fail the build over a zip
            logger.warning("notebook_header: failed to build zip for %s: %s", docname, err)

    logger.info("notebook_header: wrote %d notebook download zip(s).", count)


def setup(app):
    app.connect("doctree-read", on_doctree_read)
    app.connect("build-finished", on_build_finished)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
