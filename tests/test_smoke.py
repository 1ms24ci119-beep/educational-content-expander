from app.cli import main

def test_cli_prints_expansion(capsys):
    code = main(["photosynthesis", "--subject", "biology", "--words", "5", "--seed", "2"])
    captured = capsys.readouterr()
    assert code == 0
    assert "photosynthesis" in captured.out.lower()

def test_cli_saves_output(tmp_path, capsys):
    target = tmp_path / "notes" / "expanded.txt"
    code = main(["algorithm", "--subject", "computer-science", "--words", "5",
                 "--seed", "3", "--output", str(target)])
    assert code == 0
    assert target.exists()
    assert target.read_text(encoding="utf-8").strip()
