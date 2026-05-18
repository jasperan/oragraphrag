def test_package_imports():
    import oragraphrag

    assert isinstance(oragraphrag.__version__, str)
    assert oragraphrag.__version__  # non-empty
