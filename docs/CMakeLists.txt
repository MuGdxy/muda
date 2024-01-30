# check if Doxygen is installed
find_package(Doxygen)
if (DOXYGEN_FOUND)
    # check if muda-doc submodule exists
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/muda-doc/.git")
        message(STATUS "muda-doc submodule not found: cloning it")
        
        execute_process(COMMAND git submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)

        # if clone fails, report the error
        if (NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()

    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    # put the doxygen config file in the build directory
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    # replace variable: @PROJECT_NAME@ with the project name

    # uppercase the CMAKE_PROJECT_NAME variable
    string(TOUPPER ${CMAKE_PROJECT_NAME} PROJECT_NAME)

    set(DOCXGEN_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/ ${PROJECT_SOURCE_DIR}/src/")

    # set the built document in this source directory
    set(DOXYGEN_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/muda-doc/")
    set(DOXYGEN_HTML_OUTPUT "docs")
    message(STATUS "Doxygen found: documentation will be placed at ${DOXYGEN_OUTPUT}${DOXYGEN_HTML_OUTPUT}")
    # htmls will be place at muda-doc/docs/

    set(DOCXGEN_HTML_EXTRA_FILES "")
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # note the option ALL which allows to build the docs together with the application
    add_custom_target(muda_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
    
    # copy the ./img folder to the doxygen output folder
    add_custom_command(TARGET muda_doxygen POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/img" "${DOXYGEN_OUTPUT}/docs/img"
        COMMENT "Copying images to the doxygen output folder"
        VERBATIM)
else ()
  message("Doxygen need to be installed to generate the doxygen documentation")
endif ()