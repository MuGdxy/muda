# check if Doxygen is installed
find_package(Doxygen)
if (DOXYGEN_FOUND)
    message(STATUS "Doxygen found: documentation will be placed at ${CMAKE_CURRENT_SOURCE_DIR}/")
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    # put the doxygen config file in the build directory
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    # replace variable: @PROJECT_NAME@ with the project name
    set(PROJECT_NAME ${CMAKE_PROJECT_NAME})
    set(DOCXGEN_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/ ${PROJECT_SOURCE_DIR}/src/")

    # set the built document in this source directory
    set(DOXYGEN_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/../")
    set(DOCXGEN_HTML_EXTRA_FILES "")
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # note the option ALL which allows to build the docs together with the application
    add_custom_target(muda_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
else ()
  message("Doxygen need to be installed to generate the doxygen documentation")
endif ()