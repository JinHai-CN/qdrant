#!/bin/sh

set -ex

function clear_after_tests()
{
  rm -rf .hypothesis .pytest_cache
}


# This script is supposed to be executed from the docker image

cd "$(dirname "$0")"

trap clear_after_tests EXIT

#pytest -s openapi_integration/test_nested_payload_indexing.py
pytest -s
