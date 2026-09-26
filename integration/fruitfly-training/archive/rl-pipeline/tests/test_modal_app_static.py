"""Static checks for flyfollow/rl/modal_app.py that do not import Modal or need Modal credentials."""

import ast
from pathlib import Path

MODAL_APP = Path(__file__).resolve().parent.parent / "flyfollow" / "rl" / "modal_app.py"


def load_functions():
    tree = ast.parse(MODAL_APP.read_text(encoding="utf-8"))
    functions = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
    return functions


def decorator_name(node):
    names = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Call):
            dec = dec.func
        names.append(ast.unparse(dec))
    return names


def defaults_of(node):
    args = node.args.args
    defaults = node.args.defaults
    result = {}
    offset = len(args) - len(defaults)
    for i, arg in enumerate(args):
        if i < offset:
            result[arg.arg] = "REQUIRED"
        else:
            result[arg.arg] = ast.literal_eval(defaults[i - offset])
    return result


def test_file_parses_and_has_no_em_dash():
    text = MODAL_APP.read_text(encoding="utf-8")
    ast.parse(text)
    assert chr(0x2014) not in text


def test_main_entrypoint_signature():
    main = load_functions()["main"]
    assert "app.local_entrypoint" in decorator_name(main)
    defaults = defaults_of(main)
    expected = {
        "arm": "REQUIRED",
        "seed": 1,
        "config": "configs/train_modal.yaml",
        "init_from": "",
        "push_every": 5,
        "generations": 0,
        "popsize": 0,
        "results_worktree": "",
        "run_name": "",
    }
    for name, value in expected.items():
        assert defaults[name] == value
    for arg in main.args.args:
        assert ast.unparse(arg.annotation) in ("str", "int")


def test_smoke_entrypoint_and_evaluate_function():
    functions = load_functions()
    assert "app.local_entrypoint" in decorator_name(functions["smoke"])
    assert defaults_of(functions["smoke"])["arm"] == "REQUIRED"

    evaluate = functions["evaluate"]
    assert "app.function" in decorator_name(evaluate)
    keywords = set()
    for kw in evaluate.decorator_list[0].keywords:
        keywords.add(kw.arg)
    for name in ["cpu", "memory", "timeout", "retries", "max_containers"]:
        assert name in keywords
    assert [arg.arg for arg in evaluate.args.args] == ["job"]
