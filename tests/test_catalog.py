"""Tests for the model catalog (``vhmodels.models.catalog``) with stand-in cards.

The real cards live on Hugging Face; ``collect`` takes the card reader as an
argument, so these tests run offline against the repository's manifests.
"""

from vhmodels.models import catalog
from vhmodels.models.registry import REGISTRY

COMPLETE = {
    "license": "mit",
    "modality": "proteins",
    "citation": "https://doi.org/10.0000/example",
    "developer": "Doe et al.",
    "gated": False,
}


def _fetch(cards=None):
    """A card reader returning ``cards[repo_id]``, else a complete card."""
    cards = cards or {}
    calls = []

    def fetch(repo_id):
        calls.append(repo_id)
        return dict(cards.get(repo_id, COMPLETE))

    fetch.calls = calls
    return fetch


def _model(models, project):
    return next(model for model in models if model.project == project)


def test_every_variant_has_a_card():
    for project in REGISTRY.list_models():
        for variant in REGISTRY.list_variants(project):
            assert REGISTRY.resolve(project, variant).card is not None, variant


def test_card_substitutes_the_variant_and_a_variant_card_replaces_it():
    resolved = REGISTRY.resolve("prottrans", "prot_xlnet")
    assert resolved.card.repo_id == "virtual-human-chc/prot_xlnet"
    resolved = REGISTRY.resolve("prottrans", "prot_electra_bfd")
    assert resolved.card.repo_id == "virtual-human-chc/prot_electra_discriminator_bfd"


def test_collect_reads_each_card_once():
    fetch = _fetch()
    models, problems = catalog.collect(REGISTRY, fetch)
    assert problems == []
    assert len(fetch.calls) == len(set(fetch.calls))
    dinobloom = _model(models, "dinobloom")
    assert [variant.id for variant in dinobloom.variants] == ["b", "g", "l", "s"]
    assert [variant.params for variant in dinobloom.variants] == [
        "86M",
        "1.1B",
        "304M",
        "22M",
    ]


def test_overrides_take_precedence_over_the_card():
    models, _ = catalog.collect(REGISTRY, _fetch())
    hoptimus = _model(models, "hoptimus0")
    assert hoptimus.values("modality") == ["histology"]
    assert hoptimus.values("developer") == ["Bioptimus"]
    # No override for the license: it comes from the card.
    assert hoptimus.values("license") == ["mit"]


def test_missing_fields_are_errors_for_own_cards_and_warnings_for_external_ones():
    empty = {"gated": False}
    fetch = _fetch({"virtual-human-chc/MolE": empty, "bioptimus/H-optimus-0": empty})
    _, problems = catalog.collect(REGISTRY, fetch)
    by_repo = {problem.message.split(":")[0]: problem for problem in problems}
    assert set(by_repo) == {"virtual-human-chc/MolE", "bioptimus/H-optimus-0"}
    own = by_repo["virtual-human-chc/MolE"]
    assert own.level == "error"
    assert "license, vh_modality, vh_citation, vh_developer" in own.message
    external = by_repo["bioptimus/H-optimus-0"]
    assert external.level == "warning"
    # The overrides cover modality, citation and developer.
    assert "no license;" in external.message


def test_unreadable_card_is_an_error():
    def fetch(repo_id):
        if repo_id == "virtual-human-chc/MolE":
            raise OSError("404")
        return dict(COMPLETE)

    _, problems = catalog.collect(REGISTRY, fetch)
    assert [(problem.level, problem.message) for problem in problems] == [
        ("error", "virtual-human-chc/MolE: cannot read the card (404)")
    ]


def test_unused_repos_lists_org_repos_without_a_manifest():
    repos = [
        "virtual-human-chc/DinoBloom",
        "virtual-human-chc/prot_bert_bfd_membrane",
        "virtual-human-chc/prot_electra_generator_bfd",
        "virtual-human-chc/forgotten",
    ]
    problems = catalog.unused_repos(REGISTRY, repos)
    assert [problem.message for problem in problems] == [
        "virtual-human-chc/forgotten: on Hugging Face but in no manifest"
    ]


def test_render_lists_every_model_and_variant():
    fetch = _fetch({"bioptimus/H-optimus-0": {**COMPLETE, "gated": True}})
    models, _ = catalog.collect(REGISTRY, fetch)
    text = catalog.render(models)
    for model in models:
        assert f"[{model.name}](#{model.name.lower()})" in text
        assert f"\n## {model.name}\n" in text
        for variant in model.variants:
            assert f"| `{variant.id}` |" in text
    assert text.count("**Access:** gated") == 1
    assert (
        "| `s` | 22M | [virtual-human-chc/DinoBloom]"
        "(https://huggingface.co/virtual-human-chc/DinoBloom) |"
    ) in text
