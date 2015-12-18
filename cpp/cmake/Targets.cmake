################################################################################################
# Short command for setting defeault target properties
# Usage:
#   annfab_default_properties(<target>)
function(annfab_default_properties target)
  set_target_properties(${target} PROPERTIES
    DEBUG_POSTFIX ${annfab_DEBUG_POSTFIX}
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
endfunction()

################################################################################################
# Short command for setting runtime directory for build target
# Usage:
#   annfab_set_runtime_directory(<target> <dir>)
function(annfab_set_runtime_directory target dir)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${dir}")
endfunction()
