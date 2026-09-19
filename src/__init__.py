"""Interactive open-vocabulary 3D affordance detection (PR + HCI project).

This package wraps and extends the pretrained OpenAD model. It never vendors the
upstream code: the OpenAD checkout lives in ``_ref_openad/`` (see
``scripts/setup_openad``) and is imported through :mod:`src.openad_bridge`.
"""
