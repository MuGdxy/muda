set_project("muda")

includes("xmake/options.lua")
includes("xmake/package_requires.lua")

-- **********************************
-- 
-- targets
-- 
-- **********************************

set_languages("cxx17")
add_rules("mode.debug", "mode.release")

target("muda")
    add_undefines("min","max")
    set_kind("headeronly")
    add_headerfiles("src/(muda/**.h)","src/(muda/**.inl)", {public = true})
    add_includedirs("src/", {public = true})
    if(has_config("with_check")) then
        add_defines("MUDA_CHECK_ON=1", {public = true})
    else
        add_defines("MUDA_CHECK_ON=0", {public = true})
    end
    if(has_config("with_compute_graph")) then
        add_defines("MUDA_COMPUTE_GRAPH_ON=1", {public = true})
    else
        add_defines("MUDA_COMPUTE_GRAPH_ON=0", {public = true})
    end 
    add_packages("cuda", {public = true})
    -- add_packages("eigen", {public = true})
    add_cuflags("--extended-lambda", {public = true}) -- must be set for muda
    add_cuflags("--expt-relaxed-constexpr", {public = true}) -- must be set for muda
    add_cuflags("-rdc=true", {public = true})
target_end()


-- include muda_app_base("cui") function
includes("xmake/muda_app_base.lua")

if has_config("test") then
    target("muda_unit_test")
        muda_app_base("cui")
        local test_data_dir = path.absolute("test/data")
        add_defines("unit_test_DATA_DIR=R\"(".. test_data_dir..")\"")
        add_files("test/unit_test/**.cu","test/unit_test/**.cpp")
    target_end()
    
    target("muda_eigen_test")
        muda_app_base("cui")
        add_files("test/eigen_test/**.cu","test/eigen_test/**.cpp")
    target_end()
end

if has_config("example") then
    target("muda_example")
        muda_app_base("cui")
        add_files("example/**.cu","example/**.cpp")
    target_end()
end

