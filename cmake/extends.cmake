function(target_link_libraries_if cond)
  if(cond)
    message(${ARGN})
    target_link_libraries(${ARGN})
  endif()
endfunction()