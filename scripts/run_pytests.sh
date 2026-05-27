#!/bin/bash

echo "------------------------------------------"
echo "   Running Aether Pytest Test Suite...   "
echo "------------------------------------------"

PYTHONPATH=/workspace/src python3 -m pytest -v tests/

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "!!All tests have passed!!"
fi

exit $RESULT