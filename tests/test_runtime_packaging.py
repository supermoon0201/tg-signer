from pathlib import Path


def test_default_project_dep_uses_headless_opencv():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"opencv-python-headless"' in pyproject
    assert '"opencv-python",' not in pyproject


def test_default_run_dockerfile_does_not_install_gui_extra():
    dockerfile = Path("docker/Run.Dockerfile").read_text(encoding="utf-8")

    assert '".[speedup]"' in dockerfile
    assert '".[speedup,gui]"' not in dockerfile
