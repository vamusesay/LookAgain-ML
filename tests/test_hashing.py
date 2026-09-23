from lookagain_ml.hashing import digest_strings, sha256_file


def test_sha256_file_is_exact_and_stable(tmp_path):
    path = tmp_path / "value.bin"
    path.write_bytes(b"look-again")
    assert sha256_file(path) == "b2b6b0b141fa02796ace722390f74d641eff87353fbcc035fdc86e9a201e0188"
    assert sha256_file(path) == sha256_file(path)


def test_manifest_digest_is_order_sensitive():
    assert digest_strings(["a", "b"]) != digest_strings(["b", "a"])


