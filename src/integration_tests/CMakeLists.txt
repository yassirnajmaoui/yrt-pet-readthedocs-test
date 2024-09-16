project(integration_tests)

set(SOURCE_PYTEST
  test_recon.py
  test_datastruct.py
  helper.py)

if(${BUILD_PYBIND11})
  add_pytest_integration(YRTPET "${SOURCE_PYTEST}"
    ${CMAKE_SOURCE_DIR}/integration_tests
    ${CMAKE_BINARY_DIR}/integration_tests)
endif()
