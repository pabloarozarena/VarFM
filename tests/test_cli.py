from varfm import cli


def test_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "extract", lambda vcf, **kwargs: calls.append((vcf, kwargs)))
    cli.main(["variants.vcf", "--models", "evo2", "esm2", "evo2", "--out", "out", "--limit", "5"])
    cli.main(["variants.vcf"])
    assert calls == [
        ("variants.vcf", {"models": ["evo2", "esm2"], "out_dir": "out", "device": None, "limit": 5}),
        ("variants.vcf", {"models": ["esm2", "evo2"], "out_dir": "varfm_out", "device": None, "limit": None}),
    ]
