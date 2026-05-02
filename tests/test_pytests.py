def add(a,b):
  return a+b

def test_pytest_add_success():
    assert add(1, 1) == 2   

def test_pytest_add_edge_case():
    assert add(0, 0) == 0   

def test_pytest_add_negative():
    assert add(-1, -1) == -2  