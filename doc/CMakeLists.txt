# check if Doxygen is installed
find_package(Doxygen)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    # replace variable: @PROJECT_NAME@ with the project name
    set(PROJECT_NAME ${CMAKE_PROJECT_NAME})
    set(DOCXGEN_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/ ${PROJECT_SOURCE_DIR}/src/")
    set(DOCXGEN_HTML_EXTRA_FILES "")
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)


    

    # note the option ALL which allows to build the docs together with the application
    add_custom_target(muda_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    # copy ./img to ./html/img before build
    add_custom_command(TARGET muda_doxygen PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/img ${CMAKE_CURRENT_BINARY_DIR}/html/img)

        
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)