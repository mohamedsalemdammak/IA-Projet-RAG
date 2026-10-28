from src.out_of_scope_detection import (
    LayaScopeDetector,
    detect_out_of_scope,
)


class FakeRouter:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, state, questions, model, max_len):
        self.calls.append((state, questions, model, max_len))
        return self.result


def test_in_scope_result():
    router = FakeRouter({"scope": {
        "type": "choice",
        "choice": "IN_SCOPE",
        "probabilities": {"IN_SCOPE": 0.9, "OUT_OF_SCOPE": 0.05, "UNCERTAIN": 0.05},
        "answerConfidence": 0.9,
    }})
    result = detect_out_of_scope(
        "Et qui est responsable ?",
        [{"role": "user", "content": "Nous parlons des AGP."}],
        "",
        ["AGP"],
        "Documents sur les AGP",
        detector=LayaScopeDetector(router=router),
    )
    assert result["decision"] == "IN_SCOPE"
    assert router.calls[0][3] == 8192


def test_out_of_scope_result():
    router = FakeRouter({"scope": {
        "choice": "OUT_OF_SCOPE",
        "probabilities": {"IN_SCOPE": 0.0, "OUT_OF_SCOPE": 0.99, "UNCERTAIN": 0.01},
    }})
    result = detect_out_of_scope(
        "Quelle est la météo ?",
        [],
        "",
        [],
        "Documents sur les AGP",
        detector=LayaScopeDetector(router=router),
    )
    assert result["decision"] == "OUT_OF_SCOPE"


def test_invalid_model_result_becomes_uncertain():
    router = FakeRouter({"scope": {"answer": "invalid"}})
    result = detect_out_of_scope(
        "Je ne comprends pas.",
        [],
        "",
        [],
        "Documents sur les AGP",
        detector=LayaScopeDetector(router=router),
    )
    assert result["decision"] == "UNCERTAIN"
