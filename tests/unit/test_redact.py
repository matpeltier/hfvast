from hfvast.utils.redact import SecretRedactor, redact, register_secrets


def test_registered_secret_redacted():
    r = SecretRedactor()
    r.register("hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456")
    assert "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456" not in r.redact("token is hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456 ok")
    assert "***REDACTED***" in r.redact("token is hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456 ok")


def test_builtin_patterns_redacted_without_registration():
    text = "curl -H 'Authorization: Bearer hf_aaaaaaaaaaaaaaaaaaaaaaaa' https://x"
    assert "hf_aaaaaaaaaaaaaaaaaaaaaaaa" not in redact(text)
    assert "***REDACTED***" in redact(text)


def test_global_register_and_redact():
    register_secrets("sk-hfvast-abcdefghijklmnopqrs")
    assert "sk-hfvast-abcdefghijklmnopqrs" not in redact("key=sk-hfvast-abcdefghijklmnopqrs")


def test_multiple_secrets():
    r = SecretRedactor()
    r.register("secret-one-1234567890", "secret-two-0987654321")
    out = r.redact("a secret-one-1234567890 b secret-two-0987654321")
    assert "secret-one" not in out and "secret-two" not in out


def test_short_values_ignored():
    r = SecretRedactor()
    r.register("abc")  # too short to redact safely
    assert r.redact("abc") == "abc"
